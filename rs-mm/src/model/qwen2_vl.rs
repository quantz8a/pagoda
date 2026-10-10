//! Qwen2-VL image processor (also the still-image path of Qwen2.5-VL / 3-VL).
//!
//! Port of `sgl-project/sglang` `rust/sglang-mm/src/qwen_vl/mod.rs`:
//! `smart_resize` → bicubic resize → rescale + normalize → patchify into
//! `[grid_h * grid_w, C * tps * ps * ps]`, then a placeholder layout.
//! Patch geometry is [`Geometry::Grid`]`([t, h, w])` with `t = 1` for stills.
//!
//! HF flatten order: patches by `(gh/m, gw/m, m, m)`, features by
//! `(C, tps, ps, ps)`. Temporal copies of a still are duplicates.

use crate::common::{par, resize, token_layout};
use crate::pipeline::{
    DecodedImage, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, Tensor, TensorData,
    TokenLayout,
};

const MAX_RATIO: f64 = 200.0;

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// Resolved processor params. Nothing is hardcoded per checkpoint.
#[derive(Clone, Debug)]
pub struct Qwen2VlSpec {
    pub image_token_id: i64,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: Resampler,
}

/// The HF image processor the pipeline must match. Defaults to the fast path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Resampler {
    /// `Qwen2VLImageProcessor` / `…Fast` — torchvision on a uint8 tensor.
    #[default]
    AtenU8,
    /// `Qwen2VLImageProcessorPil`, behind `--disable-fast-image-processor`.
    Pil,
}

impl From<Resampler> for resize::Resample {
    fn from(r: Resampler) -> Self {
        match r {
            Resampler::AtenU8 => resize::Resample::AtenU8,
            Resampler::Pil => resize::Resample::Pil(resize::Filter::Bicubic),
        }
    }
}

pub struct Qwen2VlProcessor {
    spec: Qwen2VlSpec,
    /// Per-channel u8 → normalized-f32 lookup; see [`normalize_lut`].
    lut: [[f32; 256]; 3],
}

/// `1 / rescale_factor`.
const INV_RESCALE: f32 = 255.0;

/// u8 → normalized f32, rounded as the mirrored processor rounds. The slow one
/// rescales then normalizes; the fast one folds the rescale into mean/std first,
/// which differs on 128 of the 256 inputs when mean = std = 0.5.
fn normalize_lut(resample: Resampler, mean: f32, std: f32) -> [f32; 256] {
    match resample {
        Resampler::Pil => core::array::from_fn(|v| (v as f32 / INV_RESCALE - mean) / std),
        Resampler::AtenU8 => {
            let (mean, std) = (mean * INV_RESCALE, std * INV_RESCALE);
            core::array::from_fn(|v| (v as f32 - mean) / std)
        }
    }
}

impl Qwen2VlProcessor {
    pub fn new(spec: Qwen2VlSpec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("qwen2_vl spec: sizes must be positive".into());
        }
        let lut = core::array::from_fn(|c| {
            normalize_lut(spec.resample, spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self { spec, lut })
    }

    /// HF flatten: patches ordered `(gh/m, gw/m, m, m)`, features `(C, tps, ps, ps)`.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        let (ps, m, tps) = (
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
        );
        let (gh, gw) = (h / ps, w / ps);
        let dim = 3 * tps * ps * ps;
        let block_row = gw * m * dim;
        let mut out = vec![0.0f32; gh * gw * dim];

        par::for_chunks_mut(&mut out, block_row, |i, chunk| {
            let mut p = 0;
            for j in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let y0 = (i * m + mh) * ps;
                        let x0 = (j * m + mw) * ps;
                        let patch = &mut chunk[p * dim..(p + 1) * dim];
                        for c in 0..3 {
                            let ch = &mut patch[c * tps * ps * ps..];
                            for py in 0..ps {
                                let src = ((y0 + py) * w + x0) * 3 + c;
                                for px in 0..ps {
                                    ch[py * ps + px] = self.lut[c][rgb[src + px * 3] as usize];
                                }
                            }
                            let (t0, rest) = ch.split_at_mut(ps * ps);
                            for t in 0..tps - 1 {
                                rest[t * ps * ps..(t + 1) * ps * ps].copy_from_slice(t0);
                            }
                        }
                        p += 1;
                    }
                }
            }
        });
        out
    }

    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }
}

impl MmFamilyProcessor for Qwen2VlProcessor {
    fn process_item(&self, media: &DecodedImage) -> Result<ProcessedItem, String> {
        let (h, w) = (media.height, media.width);
        let grid = grid_split(
            h,
            w,
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let (th, tw) = (
            grid[1] as usize * self.spec.patch_size,
            grid[2] as usize * self.spec.patch_size,
        );
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = resize::resize_rgb(&media.rgb, h, w, th, tw, self.spec.resample.into());
            &resized
        } else {
            media.rgb.as_slice()
        };
        let (gh, gw) = (grid[1] as usize, grid[2] as usize);
        let pixel_values = self.patchify(data, th, tw);
        let dim = pixel_values.len() / (gh * gw);
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![gh * gw, dim],
                data: TensorData::F32(pixel_values),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(vec![grid[0] as i64, grid[1] as i64, grid[2] as i64]),
                },
            )],
            geometry: Geometry::Grid(grid),
        })
    }

    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String> {
        let counts = items
            .iter()
            .map(|geometry| self.tokens_per_image(&geometry.grid()))
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let mrope_items = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), geometry)| MropeItem {
                start,
                end,
                grid: geometry.grid(),
            })
            .collect::<Vec<_>>();
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// Split one still into a `[t, h, w]` patch grid (`t = 1`).
///
/// `h` and `w` are patch counts after [`smart_resize`], so both are multiples
/// of `merge_size`. This is the geometry [`MmFamilyProcessor::process_item`]
/// stores as [`Geometry::Grid`].
pub fn grid_split(
    height: usize,
    width: usize,
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<[u32; 3], String> {
    if patch_size == 0 || merge_size == 0 {
        return Err("qwen2_vl grid_split: sizes must be positive".into());
    }
    let factor = patch_size * merge_size;
    let (th, tw) = smart_resize(height, width, factor, min_pixels, max_pixels)?;
    let (gh, gw) = (th / patch_size, tw / patch_size);
    if gh == 0 || gw == 0 || gh % merge_size != 0 || gw % merge_size != 0 {
        return Err(format!(
            "qwen2_vl: patch grid {gh}x{gw} is empty or not a multiple of merge_size {merge_size}"
        ));
    }
    Ok([1, gh as u32, gw as u32])
}

/// Python-`round()` (round-half-to-even), which `round_by_factor` relies on.
fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        (x / 2.0).round() * 2.0
    } else {
        x.round()
    }
}

/// Qwen `smart_resize`: dims divisible by `factor`, total pixels within
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let (h, w) = (height as f64, width as f64);
    if height == 0 || width == 0 {
        return Err("empty image".into());
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        return Err(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        ));
    }
    let f = factor as f64;
    let mut h_bar = ((round_half_even(h / f) * f) as usize).max(factor);
    let mut w_bar = ((round_half_even(w / f) * f) as usize).max(factor);
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f) as usize;
        w_bar = ((w / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    if h_bar == 0 || w_bar == 0 {
        return Err(format!(
            "smart_resize: {height}x{width} degenerates to {h_bar}x{w_bar} at \
             max_pixels={max_pixels}; image is too thin for this pixel budget"
        ));
    }
    Ok((h_bar, w_bar))
}

/// Image-only M-RoPE fast path. Text runs sequentially on all three rows;
/// each image spans `(t, h/m, w/m)` index grids; positions advance by
/// `max(t, h/m, w/m)` past an image. Returns flattened row-major
/// `[3, input_len]` positions and the delta (`max + 1 - input_len`).
pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64), String> {
    let len = input_len;
    let mut pos = vec![0i64; 3 * len];
    let fill_text = |st: usize, n: usize, base: i64, pos: &mut [i64]| {
        for k in 0..n {
            let v = base + k as i64;
            pos[st + k] = v;
            pos[len + st + k] = v;
            pos[2 * len + st + k] = v;
        }
    };
    let mut st = 0usize;
    let mut next_pos = 0i64;
    for item in items {
        let (start, end) = (item.start as usize, item.end as usize);
        if start < st || end >= len {
            return Err(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            ));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;

        let t = item.grid[0] as usize;
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if t * gh * gw != end - start + 1 {
            return Err("mrope: token span does not match grid".into());
        }
        for ti in 0..t {
            for hi in 0..gh {
                for wi in 0..gw {
                    let idx = start + (ti * gh + hi) * gw + wi;
                    pos[idx] = next_pos + ti as i64;
                    pos[len + idx] = next_pos + hi as i64;
                    pos[2 * len + idx] = next_pos + wi as i64;
                }
            }
        }
        next_pos += (t.max(gh).max(gw)) as i64;
        st = end + 1;
    }
    if st < len {
        fill_text(st, len - st, next_pos, &mut pos);
    }
    let max = pos.iter().copied().max().unwrap_or(-1);
    Ok((pos, max + 1 - len as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::token_layout::apply_layout;
    use crate::pipeline::TensorData;

    fn spec() -> Qwen2VlSpec {
        Qwen2VlSpec {
            image_token_id: 1,
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            resample: Resampler::default(),
        }
    }

    /// Qwen2-VL patch grid against the Python `smart_resize` numbers locked in
    /// `sglang-mm`. `factor = patch_size * merge_size` (14 * 2 = 28).
    #[test]
    fn grid_split_matches_reference() {
        // 1365x2048 → 1372x2044 → 98 x 146 patches.
        assert_eq!(
            grid_split(1365, 2048, 14, 2, 3136, 12_845_056).unwrap(),
            [1, 98, 146]
        );
        // 100x100 up to the factor → 112x112 → 8 x 8.
        assert_eq!(
            grid_split(100, 100, 14, 2, 3136, 12_845_056).unwrap(),
            [1, 8, 8]
        );
        // Downscale: 4000x3000 exceeds 1280*28*28 → 840x1148 → 60 x 82.
        assert_eq!(
            grid_split(3000, 4000, 14, 2, 3136, 1_003_520).unwrap(),
            [1, 60, 82]
        );
        // Upscale below min_pixels → 56x56 → 4 x 4.
        assert_eq!(
            grid_split(20, 20, 14, 2, 3136, 12_845_056).unwrap(),
            [1, 4, 4]
        );
        // Qwen3.5 factor 32 (patch 16 * merge 2): 1365x2048 → 1376x2048 → 86 x 128.
        assert_eq!(
            grid_split(1365, 2048, 16, 2, 65536, 16_777_216).unwrap(),
            [1, 86, 128]
        );
        // Banker's rounding tie: 48/32 = 1.5 rounds to 2. Grid 250 x 4.
        assert_eq!(
            grid_split(4000, 48, 16, 2, 4, 1 << 30).unwrap(),
            [1, 250, 4]
        );
        assert!(grid_split(10000, 10, 14, 2, 3136, 12_845_056).is_err());
    }

    #[test]
    fn process_item_geometry_is_grid() {
        let proc = Qwen2VlProcessor::new(spec()).unwrap();
        // 4x8 is already divisible by patch*merge = 4, and above min_pixels.
        let item = proc
            .process_item(&DecodedImage {
                rgb: vec![0u8; 4 * 8 * 3],
                height: 4,
                width: 8,
            })
            .unwrap();
        assert_eq!(item.geometry, Geometry::Grid([1, 2, 4]));
        match item.feature.data {
            TensorData::F32(values) => {
                // gh * gw = 8 patches, dim = 3 * tps * ps * ps = 24.
                assert_eq!(item.feature.shape, vec![8, 24]);
                assert_eq!(values.len(), 8 * 24);
            }
            TensorData::I64(_) => panic!("pixel_values must be f32"),
        }
        let layout = proc.layout(&[7, 1, 9], &[item.geometry]).unwrap();
        let expanded = apply_layout(&[7, 1, 9], &layout, 1).unwrap();
        // tokens = 1*2*4 / 4 = 2 placeholder copies.
        assert_eq!(expanded.input_ids, vec![7, 1, 1, 9]);
        assert_eq!(expanded.offsets, vec![(1, 2)]);
    }

    #[test]
    fn normalize_lut_differs_per_resampler() {
        let pil = normalize_lut(Resampler::Pil, 0.5, 0.5);
        let aten = normalize_lut(Resampler::AtenU8, 0.5, 0.5);
        assert_eq!(pil.iter().zip(aten).filter(|(p, a)| *p != a).count(), 128);
        for lut in [pil, aten] {
            assert_eq!(lut[0], -1.0);
            assert_eq!(lut[255], 1.0);
        }
    }

    #[test]
    fn smart_resize_matches_python_reference() {
        assert_eq!(
            smart_resize(1365, 2048, 28, 3136, 12845056).unwrap(),
            (1372, 2044)
        );
        assert_eq!(
            smart_resize(100, 100, 28, 3136, 12845056).unwrap(),
            (112, 112)
        );
        assert_eq!(
            smart_resize(3000, 4000, 28, 3136, 1003520).unwrap(),
            (840, 1148)
        );
        assert_eq!(smart_resize(20, 20, 28, 3136, 12845056).unwrap(), (56, 56));
        assert_eq!(
            smart_resize(1365, 2048, 32, 65536, 16777216).unwrap(),
            (1376, 2048)
        );
        assert_eq!(smart_resize(4000, 48, 32, 4, 1 << 30).unwrap(), (4000, 64));
        assert!(smart_resize(10000, 10, 28, 3136, 12845056).is_err());
    }

    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        assert!(smart_resize(10, 2000, 28, 3136, 3136).is_err());

        let mut spec = spec();
        spec.patch_size = 14;
        spec.min_pixels = 3136;
        spec.max_pixels = 3136;
        let proc = Qwen2VlProcessor::new(spec).unwrap();
        let err = proc
            .process_item(&DecodedImage {
                rgb: vec![0u8; 10 * 2000 * 3],
                height: 10,
                width: 2000,
            })
            .err()
            .expect("degenerate geometry must be an Err, never a panic");
        assert!(err.contains("smart_resize"), "unexpected error: {err}");
    }

    #[test]
    fn patchify_layout_matches_hf_order() {
        let (h, w) = (4usize, 8usize);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = (y * 16 + x * 2 + c) as u8;
                }
            }
        }
        let proc = Qwen2VlProcessor::new(spec()).unwrap();
        let pv = proc.patchify(&rgb, h, w);
        let dim = 24;
        assert_eq!(pv.len(), 2 * 4 * dim);

        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        assert_eq!(pv[dim], lut(0, 2, 0));
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        let ps2 = 4;
        assert_eq!(pv[dim + ps2], pv[dim]);
        assert_eq!(pv[2 * ps2], lut(0, 0, 1));
    }

    #[test]
    fn mrope_image_only_matches_reference() {
        let items = [MropeItem {
            start: 3,
            end: 8,
            grid: [1, 4, 6],
        }];
        let (pos, delta) = mrope_image_only(11, &items, 2).unwrap();
        let len = 11;
        for k in 0..3 {
            assert_eq!(
                (pos[k], pos[len + k], pos[2 * len + k]),
                (k as i64, k as i64, k as i64)
            );
        }
        assert_eq!((pos[3], pos[len + 3], pos[2 * len + 3]), (3, 3, 3));
        assert_eq!((pos[4], pos[len + 4], pos[2 * len + 4]), (3, 3, 4));
        assert_eq!((pos[6], pos[len + 6], pos[2 * len + 6]), (3, 4, 3));
        assert_eq!((pos[9], pos[len + 9], pos[2 * len + 9]), (6, 6, 6));
        assert_eq!((pos[10], pos[len + 10], pos[2 * len + 10]), (7, 7, 7));
        assert_eq!(delta, -3);
    }
}

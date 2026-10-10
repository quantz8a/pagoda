//! Family seam for a decoded image: the processor produces tensors and a
//! [`Geometry`], and describes prompt expansion as a [`TokenLayout`].

/// Typed tensor payload produced by a family.
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

pub type NamedTensors = Vec<(String, Tensor)>;

/// HWC u8 RGB handed to [`MmFamilyProcessor::process_item`].
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

/// Family-internal geometry of one processed item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    /// `[t, h, w]` patch grid. Stills use `t = 1`.
    Grid([u32; 3]),
}

impl Geometry {
    pub fn grid(self) -> [u32; 3] {
        match self {
            Geometry::Grid(grid) => grid,
        }
    }
}

/// One processed image: feature tensor, auxiliary tensors, geometry.
pub struct ProcessedItem {
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Geometry,
}

/// Tokens one media item occupies in the expanded prompt.
pub enum TokenPattern {
    /// N copies of one placeholder id (Qwen-style).
    Repeat { id: i64, n: usize },
    /// Explicit id sequence (tile markers, row separators).
    Explicit(Vec<i64>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy `src` (a range into the original ids) verbatim.
    Text(std::ops::Range<usize>),
    /// Media item `item`'s token span.
    Media { item: usize, pattern: TokenPattern },
}

/// Prompt geometry as data. [`crate::common::token_layout::apply_layout`]
/// expands it.
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// Position scheme of the expanded prompt.
pub enum PositionOutput {
    Rope1D,
    /// Flattened row-major `[3, input_len]` positions and `max + 1 - input_len`.
    MRope {
        positions: Vec<i64>,
        delta: i64,
    },
}

/// Per-model hooks. Parameters come from the caller-supplied spec.
pub trait MmFamilyProcessor {
    fn process_item(&self, media: &DecodedImage) -> Result<ProcessedItem, String>;

    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String>;

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}

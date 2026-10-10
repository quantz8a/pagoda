//! Token-layout mechanics. Families describe prompt geometry as a
//! [`TokenLayout`]; [`apply_layout`] applies it.

use crate::pipeline::{Segment, TokenLayout, TokenPattern};

/// Expanded prompt plus, per media item, the inclusive `(start, end)` range.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i64>,
    pub offsets: Vec<(u32, u32)>,
}

/// Apply a family's [`TokenLayout`] to the original prompt.
///
/// Text ranges must be in bounds, ascending, and cover every source token
/// exactly once together with the media placeholders. Every item is placed
/// exactly once, and no item expands to zero tokens.
pub fn apply_layout(
    src: &[i64],
    layout: &TokenLayout,
    n_items: usize,
) -> Result<ExpandedPrompt, String> {
    let mut out = Vec::new();
    let mut offsets: Vec<Option<(u32, u32)>> = vec![None; n_items];
    let mut consumed = 0usize;
    for segment in &layout.segments {
        match segment {
            Segment::Text(range) => {
                let text = src
                    .get(range.clone())
                    .ok_or_else(|| format!("layout: text range {range:?} out of bounds"))?;
                if range.start != consumed {
                    return Err(format!(
                        "layout: text range {range:?} does not resume at source index {consumed}"
                    ));
                }
                consumed = range.end;
                out.extend_from_slice(text);
            }
            Segment::Media { item, pattern } => {
                if consumed >= src.len() {
                    return Err(format!(
                        "layout: media item {item} has no source placeholder at index {consumed}"
                    ));
                }
                consumed += 1;
                let start = out.len() as u32;
                let n = match pattern {
                    TokenPattern::Repeat { id, n } => {
                        out.resize(out.len() + n, *id);
                        *n
                    }
                    TokenPattern::Explicit(ids) => {
                        out.extend_from_slice(ids);
                        ids.len()
                    }
                };
                if n == 0 {
                    return Err(format!("layout: media item {item} expands to zero tokens"));
                }
                let slot = offsets
                    .get_mut(*item)
                    .ok_or_else(|| format!("layout: media item {item} out of range"))?;
                if slot.replace((start, start + n as u32 - 1)).is_some() {
                    return Err(format!("layout: media item {item} placed twice"));
                }
            }
        }
    }
    if consumed != src.len() {
        return Err(format!(
            "layout: covers {consumed} of {} source token(s)",
            src.len()
        ));
    }
    let offsets = offsets
        .into_iter()
        .enumerate()
        .map(|(i, slot)| slot.ok_or_else(|| format!("layout: media item {i} not placed")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExpandedPrompt {
        input_ids: out,
        offsets,
    })
}

/// Each occurrence of `placeholder_id` becomes `counts[i]` copies.
pub fn layout_by_placeholder(
    ids: &[i64],
    placeholder_id: i64,
    counts: &[usize],
) -> Result<TokenLayout, String> {
    let found = ids.iter().filter(|&&id| id == placeholder_id).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} media placeholder(s) but {} media item(s)",
            counts.len()
        ));
    }
    let mut segments = Vec::new();
    let mut text_start = 0;
    let mut item = 0;
    for (pos, &id) in ids.iter().enumerate() {
        if id == placeholder_id {
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            segments.push(Segment::Media {
                item,
                pattern: TokenPattern::Repeat {
                    id: placeholder_id,
                    n: counts[item],
                },
            });
            item += 1;
            text_start = pos + 1;
        }
    }
    if text_start < ids.len() {
        segments.push(Segment::Text(text_start..ids.len()));
    }
    Ok(TokenLayout { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(ids: &[i64], placeholder: i64, counts: &[usize]) -> Result<ExpandedPrompt, String> {
        apply_layout(
            ids,
            &layout_by_placeholder(ids, placeholder, counts)?,
            counts.len(),
        )
    }

    #[test]
    fn expands_in_order_with_inclusive_offsets() {
        let e = expand(&[7, 1, 8, 1, 9], 1, &[2, 3]).unwrap();
        assert_eq!(e.input_ids, vec![7, 1, 1, 8, 1, 1, 1, 9]);
        assert_eq!(e.offsets, vec![(1, 2), (4, 6)]);
    }

    #[test]
    fn count_mismatch_errs() {
        assert!(expand(&[7, 1, 9], 1, &[2, 3]).is_err());
        assert!(expand(&[7, 1, 1, 9], 1, &[2]).is_err());
    }

    #[test]
    fn zero_count_errs() {
        assert!(expand(&[7, 1, 9], 1, &[0]).is_err());
    }

    #[test]
    fn no_placeholders_no_items_ok() {
        let e = expand(&[7, 8], 1, &[]).unwrap();
        assert_eq!(e.input_ids, vec![7, 8]);
        assert!(e.offsets.is_empty());
    }
}

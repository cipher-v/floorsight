# Privacy cross-check, AWS run 3 (23 Oct 2026), with a manual review of every flagged frame type

What was measured: for every person the earlier (AGPL, since replaced) detector found, the share of their estimated head
region that the new Apache-2.0 privacy stage pixelated. A head counts as covered at 80 % or more. Boxes only; no video
is read. Receipt: `privacy-crosscheck-aws-run3-20261023.json`.

| Clip | Reference heads | Covered | Frames with a head under 80 % |
|---|---|---|---|
| screen-002 | 196 | 100 % | 0 |
| screen-032 | 352 | 98.9 % | 4 |
| clip-dev-004 | 602 | 98.8 % | 7 |
| clip-dev-002 | 429 | 97.7 % | 7 |
| screen-033 | 1057 | 96.9 % | 27 |
| **All five** | **2636** | **98.0 % (2582)** | **45** |

clip-dev-003 has no reference run, so it is not in the cross-check.

Manual review (24 Oct): montages of the flagged people were cut from the privacy-filtered video on AWS (no raw
pixels) and looked at one by one: 27 frames of screen-033, 6 of 7 in clip-dev-004, 5 of 7 in clip-dev-002, 2 of 4 in
screen-032.

- In almost every flagged frame the head is pixelated. The old detector's box sat lower (a bending or partly hidden
  person), so the estimated head region missed the mosaic.
- One real gap: clip-dev-002, frames 206 and 211. The top of one person's head, about 20 px across, is unpixelated
  behind a machine panel. No face is visible and the person is not identifiable at this size. Those frames are not
  shown enlarged in the demo video.

Fail-closed test on the same run: 12 frames encoded, frames 3 and 7 forced to fail, both written black and listed as
withheld (pass). The target we set ourselves was 99 % coverage; the run reached 98 %, and we report that rather than
the target.

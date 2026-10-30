# PROGRESS

goal      candidate for 'attention' that matches the reference and runs >= 6.66x faster
best      attempt #2, 5.22x (Switch the dots to tensor-core `tf32x3` (fp32-accurate), use 64x64 tiles, and launch the heaviest causal query blocks first for load balance.)

## tried
- #1 [speed] Bigger 64x32 tiles with the loop split into unmasked off-diagonal blocks and masked diagonal blocks, plus exp2 softmax, so K/V tiles are reused across 64 query rows.: 5.07x (2.542 ms vs 12.888 ms)
- #2 [speed] Switch the dots to tensor-core `tf32x3` (fp32-accurate), use 64x64 tiles, and launch the heaviest causal query blocks first for load balance.: 5.22x (2.494 ms vs 13.007 ms)

## known
- Strategy for this step: Tile the computation so each block reuses data from registers or shared memory instead of re-reading global memory; pick tile shapes that fit the hardware.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file

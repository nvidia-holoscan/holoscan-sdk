# Third-party dependencies

This directory contains CMake definitions for third-party dependencies used by
Holoscan SDK. Each definition records the public upstream source, pinned
revision, and any ordered patches applied from `patches/<dependency>/`.

Carried patches normally describe their purpose and record an
`Upstream-Status:` disposition following the
[Yocto Project convention](https://docs.yoctoproject.org/contributor-guide/recipe-style-guide.html#patch-upstream-status).
Keeping filenames and ordering stable when the logical change remains the same
helps make patch-series updates easier to review.

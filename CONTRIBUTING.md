# Contribute to Holoscan SDK

Welcome to Holoscan SDK! We're glad that you're considering contributing to the platform.

Holoscan SDK is released on GitHub as open source software to better support the community and facilitate feedback. By
contributing you agree to follow our [code of conduct](./CODE_OF_CONDUCT.md).

## Reporting Feedback

Community feedback helps us improve the Holoscan SDK platform to better meet user needs. We use GitHub [Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues) to track feedback and problems over time, as well as to provide
limited support to Holoscan SDK users.

Consider reviewing existing issues or opening a new issue if you:

- Have a question about using a Holoscan SDK feature
- Notice errors or unexpected behavior coming from Holoscan SDK
- Have an idea for a change that might benefit other users

When reporting an error, please include relevant details that will help our team investigate the issue. Details might include:

- A summary of the problem
- The behavior you have observed
- The behavior you expected
- Details about your PC, including the architecture (x86_64 or arm64) and GPU
- The version of Holoscan SDK where you observed the problem
- Any relevant logs or images to help investigate the issue

You can also refer to the Holoscan SDK [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/robotics-edge-computing/holoscan/757) for support questions and community discussions.

## Suggesting Changes

Users are welcome to suggest code changes to Holoscan SDK in the form of [Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues) or [Pull Requests](https://github.com/nvidia-holoscan/holoscan-sdk/pulls).

### Issues

Please open a new [issue](https://github.com/nvidia-holoscan/holoscan-sdk/issues) if you'd like to request a new feature
or propose a feature design. You can tag a specific community maintainer in your post with "@", or we'll update
when we've had a chance to review your post.

### Pull Requests

While we primarily develop Holoscan SDK internally, we also accept external contributions that help move the platform
forward. We typically favor contributions that aim to fix an existing issue or improve documentation, but we'll also integrate new features
and enhancements when they're aligned with the Holoscan SDK vision. Please check in with the development team to propose your idea before spending time working on new features. This will help prevent duplicate effort and avoid spending time on features that would be unlikely to be merged. We'll typically refer new operators for contribution
to the downstream [HoloHub](https://github.com/nvidia-holoscan/holohub) community project.

External pull requests that add, remove, or modify files under [`skills/`](./skills/README.md) are not accepted
while that collection is curated by the Holoscan team. Route skill ideas through the forum link above.
The rest of the SDK continues to welcome community contributions as described in this guide.

You might contribute to Holoscan to help us address fixes earlier in our development cycle, or to suggest improvements to Holoscan SDK that you believe would broadly benefit the Holoscan community.

Holoscan SDK follows a monthly release process that includes internal quality assurance. To add a fix or feature,
we request that you [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) Holoscan SDK, develop in a branch, and submit your change as a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) against the latest Holoscan SDK release commit. If we accept your submission after external discussion, we will integrate those changes within our internal development and credit you in Git commit history. Any changes that we accept from community contributions will undergo quality assurance testing before they are included in the next Holoscan SDK release.

**Note**:  We recommend that new GitHub users read GitHub's [Getting Started](https://docs.github.com/en/get-started/start-your-journey) guide before opening their first pull request.

### Before you open a pull request

Run the local checklist from the SDK source root. The recommended path uses Docker and needs
GPU access through the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
./run build
./run test
./run lint
```

Enable pre-commit hooks in the host shell where you commit (not only inside Docker):

```bash
uvx pre-commit install
```

Every commit in the pull request must include a DCO sign-off (`git commit -s`). See
[Signing Off Your Work](#signing-off-your-work).

See [DEVELOP.md](./DEVELOP.md) for full build, test, and lint options, the ABI/export policy, and the pre-commit hook table.

## Signing Off Your Work

This project will only accept contributions under the Apache-2.0 license.

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
git commit -s -m "feat(core): add a new feature"
```

This will append the following to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

Use your real name and a verifiable email address. Pseudonyms and anonymous contributions are not accepted.

Full text of the DCO (<https://developercertificate.org/>):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Tracking Development

We take all community feedback into consideration. If we don't believe a proposed change aligns with our direction
for Holoscan SDK, or if we don't expect we can prioritize a task within a reasonable timeframe, we'll let you
know by appropriately labeling or closing the issue or pull request with an explanatory comment.

For items that we do plan to pursue or integrate, we use
[GitHub Milestones](https://github.com/nvidia-holoscan/holoscan-sdk/milestones)
to communicate our release planning. We will add community issues and pull requests to the approximate monthly release
milestone when we expect to pursue that development. Issues that we mark as "needs triage" are not part of a milestone
and will be revisited once each month to determine our priorities. See [ROADMAP.md](./ROADMAP.md) for a
user-facing overview of this process and the 5.x direction.

## Header Guards and ABI Exports

Contributors changing Holoscan Core or SDK-owned C++ libraries should follow the
ABI/export policy in [`DEVELOP.md`](./DEVELOP.md#abi-visibility-and-header-guards).
Headers use `#pragma once`, public DSO ABI uses generated export headers, and
`detail/` paths are internal.

HoloMQ is an SDK-owned component: build from the SDK root with `./run build` and
link `holoscan::holomq` (or compatibility target `holomq::holomq`) when using its
APIs directly. Its ABI and export policy lives in
[`docs/holomq/abi_and_exports.md`](./docs/holomq/abi_and_exports.md).

## Additional Information

Please refer to the project [README](./README.md) document for additional developer information. Happy coding!

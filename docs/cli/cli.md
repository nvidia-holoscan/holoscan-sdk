(holoscan-cli)=

# Holoscan CLI

`holoscan` - a command-line interface for packaging and running your Holoscan applications into [HAP-compliant](./hap.md) containers.

## Synopsis

`holoscan` [](#cli-help) [](#cli-log-level) {[package](./package.md),[run](./run.md),[version](./version.md)}

## Positional Arguments

<!-- We add a hidden toctree with links to the subcommand files
with the syntax `title <file_name>` to edit the title in the
toctree to the command name only -->
```{toctree}
:maxdepth: 1
:hidden:

package <package>
run <run>
version <version>
```

<!-- We use h3 instead of ### to not include in toctree -->
[<h3>`package`</h3>](./package.md)

Package a Holoscan application

[<h3>`run`</h3>](./run.md)

Run a packaged Holoscan application

[<h3>`version`</h3>](./version.md)

Print version information for the Holoscan SDK

[<h3>`nics`</h3>](./nics.md)

Print all available network interface cards and its assigned IP address

## CLI-Wide Flags

(#cli-help)=

### `[--help|-h]`

Display detailed help.

(#cli-log-level)=

### `[--log-level|-l {DEBUG,INFO,WARN,ERROR,CRITICAL}]`

Override the default logging verbosity. Defaults to `INFO`.

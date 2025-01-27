(holoscan-packager)=

# Packaging Holoscan Applications

The [Holoscan App Packager](./cli/package.md), included as part of the [Holoscan CLI](./cli/cli.md) as the `package` command, allows you to package your Holoscan applications into a [HAP-compliant](./cli/hap.md) container image for distribution and deployment.

## Prerequisites

### Dependencies

Ensure the following are installed in the environment where you want to run the [CLI](./cli/cli.md):

- [**NVIDIA Container Toolkit with Docker**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
  - Developer Kits (aarch64): already included in IGX Software and JetPack
  - x86_64: tested with NVIDIA Container Toolkit 1.13.3 with Docker v24.0.1
- **Docker BuildX plugin**
  1. Check if it is installed:

     ```bash
     $ docker buildx version
     github.com/docker/buildx v0.10.5 86bdced
     ```

  2. If not, run the following commands based on the [official doc](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository):

     ```bash
     # Install Docker dependencies
     sudo apt-get update
     sudo apt-get install ca-certificates curl gnupg

     # Add Docker Official GPG Key
     sudo install -m 0755 -d /etc/apt/keyrings
     curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
     sudo chmod a+r /etc/apt/keyrings/docker.gpg

     # Configure Docker APT Repository
     echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

     # Install Docker BuildX Plugin
     sudo apt-get update
     sudo apt-get install docker-buildx-plugin
     ```

- [**QEMU**](https://github.com/multiarch/qemu-user-static) *(Optional)*
  - used for packaging container images of different architectures than the host (example: x86_64 -> arm64).

### CLI Installation

The Holoscan CLI is available as a PyPI package and can be installed with the following command:

```bash
$ pip install holoscan-cli
```

Verify the installation:

```bash
$ holoscan version
```


:::{tip}
Always install Holoscan SDK first, followed by Holoscan CLI on a system that requires both. This ensures that all necessary dependencies and packages are installed correctly, allowing for smooth operation of the Holoscan CLI.
:::



## Package an application

:::{tip}
The packager feature is also illustrated in the [cli_packager](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/cli_packager) and
[video_replayer_distributed](<https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/video_replayer_distributed>) examples.

Additional arguments are required when launching the container to enable the packaging of Holoscan applications inside the NGC Holoscan container. Please see the [NGC Holoscan container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) page for additional details.
:::

1. Ensure to use the [HAP environment variables](./cli/hap.md#table-of-environment-variables) wherever possible when accessing data. For example:

   Let's take a look at the distributed video replayer example (`examples/video_replayer_distributed`).

   - **Using the Application Configuration File**

     `````{tab-set}
     ````{tab-item} C++

     In the `main` function, we call the `app->config(config_path)` function with the default configuration file.
     The `app->config(...)` checks to see if the application was executed with `--config` argument first. If `--config`
     was set, the method uses the configuration file from the `--config` argument. Otherwise, it
     checks if the environment variable `HOLOSCAN_CONFIG_PATH` is set and uses that value as the source.
     If neither were set, the default configuration file (`config_path`) is used.


     ```cpp
      int main(int argc, char** argv) {
        // Get the yaml configuration file
        auto config_path = std::filesystem::canonical(argv[0]).parent_path();
        config_path /= std::filesystem::path("video_replayer_distributed.yaml");

        auto app = holoscan::make_application<DistributedVideoReplayerApp>();
        app->config(config_path);
        app->run();

        return 0;
      }
     ```
     ````

     ````{tab-item} Python

     In the `main` function, we call the `app.config(config_file_path)` function with the default configuration file.
     The `app.config(...)` method checks to see if the application was executed with `--config` argument first. If `--config`
     was set, the method uses the configuration file from the `--config` argument. Otherwise, it
     checks if the environment variable `HOLOSCAN_CONFIG_PATH` is set and uses that value as the source.
     If neither were set, the default configuration file (`config_file_path`) is used.

     ```python
      def main():
          input_path = get_input_path()
          config_file_path = os.path.join(os.path.dirname(__file__), "video_replayer_distributed.yaml")

          logging.info(f"Reading application configuration from {config_file_path}")

          app = DistributedVideoReplayerApp(input_path)
          app.config(config_file_path)
          app.run()
     ```
     `````

   - **Using Environment Variable `HOLOSCAN_INPUT_PATH` for Data Input**

     `````{tab-set}
     ````{tab-item} C++

     In `Fragment1`, we try to set the input video directory with the value defined in
     `HOLOSCAN_INPUT_PATH`. When we instantiate a new Video Stream Replayer operator, we pass in all
     configuration values from the `from_config("replayer")` call. In addition, we include `args` that
     we created with the value from `HOLOSCAN_INPUT_PATH` if available as the last argument to
     override the `directory` setting.

     ```cpp
     class Fragment1 : public holoscan::Fragment {
       public:
         void compose() override {
           using namespace holoscan;
           ArgList args;
           auto data_directory = std::getenv("HOLOSCAN_INPUT_PATH");
           if (data_directory != nullptr && data_directory[0] != '\0') {
             auto video_directory = std::filesystem::path(data_directory);
             video_directory /= "racerx";
             args.add(Arg("directory", video_directory.string()));
             HOLOSCAN_LOG_INFO("Using video from {}", video_directory.string());
           }
           auto replayer =
               make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);
           add_operator(replayer);
         }
     };
     ```
     ````

     ````{tab-item} Python

     In `Fragment1`, we try to set the input video directory with the value defined in
     `HOLOSCAN_INPUT_PATH`. When we instantiate a new Video Stream Replayer operator, we pass in
     the `video_path` along with all `replayer` configurations found in the configuration file.

     ```python
     class Fragment1(Fragment):
         def __init__(self, app, name):
             super().__init__(app, name)

         def __init__(self, app, name):
             super().__init__(app, name)

         def compose(self):
             # Set the video source
             video_path = self._get_input_path()
             logging.info(
                 f"Using video from {video_path}"
             )

             # Define the replayer and holoviz operators
             replayer = VideoStreamReplayerOp(
                 self, name="replayer", directory=video_path, **self.kwargs("replayer")
             )

             self.add_operator(replayer)

         def _get_input_path(self):
             path = os.environ.get(
                 "HOLOSCAN_INPUT_PATH", os.path.join(os.path.dirname(__file__), "data")
             )
             return os.path.join(path, "racerx")
     ```
     `````

2. Include a YAML configuration file as described in the [Application Runner Configuration](./cli/run_config.md) page.

3. Use the [`holoscan package`](./cli/package.md) command to create a HAP container image. For example:

   ```bash
   holoscan package --platform x64-workstation --tag my-awesome-app --config /path/to/my/awesome/application/config.yaml /path/to/my/awesome/application/
   ```

### Additional Requirements for Non-Distributed Applications

In addition to using HAP-defined environment variables, applications must also handle parsing the `—config` argument when packaging a non-distributed application.


   - **Handle `--config` argument**

     `````{tab-set}
     ````{tab-item} C++

     Define a `parse_arguments` function to handle parsing the data path and the config file path:

     ```cpp
      bool parse_arguments(int argc, char** argv, std::string& data_path, std::string& config_path) {
        static struct option long_options[] = {
            {"data", required_argument, 0, 'd'}, {"config", required_argument, 0, 'c'}, {0, 0, 0, 0}};

        while (int c = getopt_long(argc, argv, "d:c:", long_options, NULL)) {
          if (c == -1 || c == '?') break;

          switch (c) {
            case 'c':
              config_path = optarg;
              break;
            case 'd':
              data_path = optarg;
              break;
            default:
              holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
              return false;
          }
        }

        return true;
      }
     ```

     Use the `parse_arguments()` function defined above in the `main` function:

     ```cpp
      int main(int argc, char** argv) {
      // Parse the arguments
      std::string config_path = "";
      std::string data_directory = "";
      if (!parse_arguments(argc, argv, data_directory, config_path)) { return 1; }
      if (data_directory.empty()) {
        // Get the input data environment variable
        auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
        if (input_path != nullptr && input_path[0] != '\0') {
          data_directory = std::string(input_path);
        } else {
          HOLOSCAN_LOG_ERROR(
              "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
          exit(-1);
        }
      }

      if (config_path.empty()) {
        // Get the input data environment variable
        auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
        if (config_file_path == nullptr || config_file_path[0] == '\0') {
          auto config_file = std::filesystem::canonical(argv[0]).parent_path();
          config_path = config_file / std::filesystem::path("app-config.yaml");
        } else {
          config_path = config_file_path;
        }
      }

      auto app = holoscan::make_application<App>();

      HOLOSCAN_LOG_INFO("Using configuration file from {}", config_path);
      app->config(config_path);

      HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
      app->set_datapath(data_directory);

      app->run();

      return 0;
      }
     ```
     ````

     ````{tab-item} Python

     Define a `parse_arguments` function to parse the arguments:

     ```python
      def parse_arguments() -> argparse.Namespace:
          parser = argparse.ArgumentParser(description="My Application")
          parser.add_argument(
              "--data",
              type=str,
              required=False,
              default=os.environ.get("HOLOSCAN_INPUT_PATH", None),
              help="Input dataset",
          )
          parser.add_argument(
              "--config",
              type=str,
              required=False,
              default=os.environ.get(
                  "HOLOSCAN_CONFIG_PATH",
                  os.path.join(os.path.dirname(__file__), "app-config.yaml"),
              ),
              help="Application configurations",
          )

          args, _ = parser.parse_known_args()

          return args
     ```

     Parse the arguments using `parse_arguments()` defined above and start the application:
     ```python
      if __name__ == "__main__":
          args = parse_arguments()

          if args.data is None:
              logger.error(
                  "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
              )
              sys.exit(-1)

          app = MyApp(args.data)
          app.config(args.config)
          app.run()
     ```
     `````

### Common Issues When Using Holoscan Packager

#### DNS Name Resolution Error

The Holoscan Packager may be unable to resolve hostnames in specific networking environments and may show errors similar to the following:

```
curl: (6) Could not resolve host: github.com.
Failed to establish a new connection:: [Errno -3] Temporary failure in name solution...
```

To resolve these errors, edit the `/etc/docker/daemon.json` file to include `dns` and `dns-serach` fields as follows:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "dns": ["IP-1", "IP-n"],
    "dns-search": ["DNS-SERVER-1", "DNS-SERVER-n"]
}
```

You may need to consult your IT team and replace `IP-x` and `DNS-SERVER-x` with the provided values.

## Run a packaged application

The packaged Holoscan application container image can run with the [Holoscan App Runner](./cli/run.md):

```bash
holoscan run -i /path/to/my/input -o /path/to/application/generated/output my-application:1.0.1
```

Since the packaged Holoscan application container images are OCI-compliant, they're also compatible with [Docker](https://www.docker.com), [Kubernetes](https://kubernetes.io/), and [containerd](https://containerd.io/).

Each packaged Holoscan application container image includes tools inside for extracting the
embedded application, manifest files, models, etc.  To access the tool and to view all available
options, run the following:

```bash
docker run -it my-container-image[:tag] help
```

The command should prints following:

```bash
USAGE: /var/holoscan/tools [command] [arguments]...
 Command List
    extract  ---------------------------  Extract data based on mounted volume paths.
        /var/run/holoscan/export/app        extract the application
        /var/run/holoscan/export/config     extract app.json and pkg.json manifest files and application YAML.
        /var/run/holoscan/export/models     extract models
        /var/run/holoscan/export/docs       extract documentation files
        /var/run/holoscan/export            extract all of the above
        IMPORTANT: ensure the directory to be mounted for data extraction is created first on the host system.
                   and has the correct permissions. If the directory had been created by the container previously
                   with the user and group being root, please delete it and manually create it again.
    show  -----------------------------  Print manifest file(s): [app|pkg] to the terminal.
        app                                 print app.json
        pkg                                 print pkg.json
    env  -------------------------  Print all environment variables to the terminal.
```

:::{note}
The tools can also be accessed inside the Docker container via `/var/holoscan/tools`.
:::

For example, run the following commands to extract the manifest files and the application configuration file:

```bash
# create a directory on the host system first
mkdir -p config-files

# mount the directory created to /var/run/holoscan/export/config
docker run -it --rm -v $(pwd)/config-files:/var/run/holoscan/export/config my-container-image[:tag] extract

# include -u 1000 if the above command reports a permission error
docker run -it --rm -u 1000 -v $(pwd)/config-files:/var/run/holoscan/export/config my-container-image[:tag] extract

# If the permission error continues to occur, please check if the mounted directory has the correct permission.
# If it doesn't, please recreate it or change the permissions as needed.

# list files extracted
ls config-files/

# output:
# app.json  app.yaml  pkg.json
```

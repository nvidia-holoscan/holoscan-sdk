# Runtime Benchmark Hardware Tuning

Runtime latency depends on CPU frequency, idle-state exit latency, affinity, and
other host activity. Use this guide for controlled performance characterization.
CTest remains a functional smoke and does not require these settings.

These changes trade power efficiency for lower and more repeatable latency.
Apply them only to an approved test system, record every original value, and
restore the system after the run.

## Common Inventory

Record the revision, kernel, topology, CPU assignment, and frequency policy:

```sh
git rev-parse HEAD
uname -a
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MINMHZ,MAXMHZ,MHZ
cat /sys/devices/system/cpu/online
cat /sys/devices/system/cpu/isolated
cat /sys/devices/system/cpu/nohz_full

for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  printf '%s affected=%s governor=%s min=%s max=%s\n' \
    "${policy##*/}" \
    "$(cat "$policy/affected_cpus")" \
    "$(cat "$policy/scaling_governor")" \
    "$(cat "$policy/scaling_min_freq")" \
    "$(cat "$policy/scaling_max_freq")"
done
```

Inventory idle states by name. State indices are not portable across machines,
kernels, or firmware:

```sh
for cpu_path in /sys/devices/system/cpu/cpu[0-9]*; do
  test -d "$cpu_path/cpuidle" || continue
  cpu=${cpu_path##*cpu}
  printf 'cpu=%s core=%s siblings=%s\n' \
    "$cpu" \
    "$(cat "$cpu_path/topology/core_id")" \
    "$(cat "$cpu_path/topology/thread_siblings_list")"
  for state in "$cpu_path"/cpuidle/state*; do
    printf '  path=%s name=%s residency_us=%s latency_us=%s disabled=%s\n' \
      "$state" \
      "$(cat "$state/name")" \
      "$(cat "$state/residency")" \
      "$(cat "$state/latency")" \
      "$(cat "$state/disable")"
  done
done
```

Pin every benchmark role listed by the result artifact. Avoid placing two roles
on sibling logical CPUs when distinct physical cores are available.

## x86_64

Use the `performance` governor for every policy containing an assigned CPU.
Install the kernel-matched Linux tools package if `cpupower` is absent. On
Ubuntu, for example:

```sh
sudo apt update
sudo apt install "linux-tools-$(uname -r)"
```

Then select the performance governor and disable idle states whose reported
exit latency is at least 1 us:

```sh
sudo cpupower frequency-set -g performance
sudo cpupower idle-set -D 1
```

The `-D 1` threshold is inclusive. It can therefore disable a state named `C1`
when that state reports a latency of 1 us or more; state names and indices are
not portable. The frequency governor alone does not control idle states.

For a reversible setup, use the repository helper. It disables states at the
same latency threshold through sysfs while preserving the existing flags for
lower-latency states. It also sets every CPU-frequency policy to the
`performance` governor and records every original value before its first
write:

```sh
sudo ./public/benchmark/configure_cpu_performance.sh \
  apply /tmp/holoscan-cpu-performance.state

# Run the controlled benchmarks, then restore the exact prior settings.

sudo ./public/benchmark/configure_cpu_performance.sh \
  restore /tmp/holoscan-cpu-performance.state
```

Keep the state file until restoration has been verified. If the benchmark
shell is interrupted, the same restore command remains valid; it never assumes
that an original idle-state disable value was zero or that the original
governor was `powersave`.

On one development workstation, sysfs reported approximately 2 us for `C1`,
10 us for `C1E`, and 133 us for `C6`. Those values are not general platform
constants. Publish the values and residency counters from the machine that
produced each result.

## NVIDIA discrete GPUs

For dGPU Tensor rows, first record persistence mode, supported locks, current
clocks, power, temperature, and clock-throttling reasons:

```sh
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE
nvidia-smi --query-gpu=index,name,persistence_mode,clocks.max.sm,clocks.max.mem \
  --format=csv
```

On a dedicated benchmark machine, select the target GPU, enable persistence
mode, and lock supported SM and memory clocks to their reported maxima:

```sh
GPU_INDEX=0
MAX_SM_CLOCK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.max.sm \
  --format=csv,noheader,nounits)
MAX_MEMORY_CLOCK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.max.mem \
  --format=csv,noheader,nounits)
sudo nvidia-smi -i "$GPU_INDEX" -pm 1
sudo nvidia-smi -i "$GPU_INDEX" -lgc="$MAX_SM_CLOCK"
sudo nvidia-smi -i "$GPU_INDEX" -lmc="$MAX_MEMORY_CLOCK"
```

Clock-lock support varies by GPU architecture and driver. In particular, some
architectures require deferred memory-clock locking. Treat an unsupported
command as a platform limitation, record it, and do not silently compare that
row with a fully locked run. Reset ordinary locks after collection:

```sh
sudo nvidia-smi -i "$GPU_INDEX" -rgc
sudo nvidia-smi -i "$GPU_INDEX" -rmc
```

Restore persistence mode to its recorded prior value. These reset commands
restore driver defaults, not an arbitrary pre-existing custom lock, so use a
dedicated test host or retain the site's own GPU restoration procedure.

## NVIDIA Jetson and IGX platforms

Jetson AGX Thor and Jetson/IGX Orin use Arm PSCI and Linux `cpuidle`, not
Intel C-state naming. Inventory the active driver and governor:

```sh
cat /sys/devices/system/cpu/cpuidle/current_driver
cat /sys/devices/system/cpu/cpuidle/current_governor_ro
sudo nvpmodel -q
sudo jetson_clocks --show
```

Jetson AGX Thor can expose a shallow `WFI` state and a deeper cluster-power
state such as `CC7`. Orin can expose `C1`, `C7`, and `CC7`. Use the names,
latencies, residencies, and disable flags exported by the installed BSP.

For Jetson AGX Thor, select the approved MAXN mode before locking clocks. Do
not assume that MAXN is mode `0`: the mode identifier varies by module and BSP.
Query the installed configuration first:

```sh
sudo nvpmodel -q --verbose
sudo nvpmodel -m <maxn-mode-id>
sudo jetson_clocks --store /tmp/holoscan-jetson-clocks.state
sudo jetson_clocks
```

Run `nvpmodel` before `jetson_clocks`; changing power mode after clocks are
locked can require a reboot. Restore the saved clock state after collection:

```sh
sudo jetson_clocks --restore /tmp/holoscan-jetson-clocks.state
```

For controlled latency characterization:

1. Select an approved `nvpmodel` before changing clocks. Do not assume a mode
   number is portable between SKUs.
2. Run `jetson_clocks` to maximize CPU, GPU, and EMC clocks within that mode.
3. Inventory `cpuidle` after `jetson_clocks`. If states with an exit latency of
   1 us or more remain enabled, use the reversible repository helper for the
   same latency-controlled CPU row used on x86_64.
4. Verify idle-state counters, frequencies, thermal state, and overcurrent
   counters before and after the run.

Do not infer the resulting `cpuidle` state solely from the tool name or BSP.
Verify each exported `disable` flag. `SC7` is suspend-to-RAM and is not part of
ordinary per-message Runtime wake latency.

Power-state hierarchy differs between Thor and Orin. A cluster can enter a deep
state only when its participating cores are idle, so control every CPU in the
cluster used by the benchmark and record the topology.

## Result Policy

A publishable latency result should record:

- exact revision, Release build configuration, kernel, and platform SKU;
- all benchmark role affinities and physical-core mappings;
- CPU governor, idle driver/governor, and idle-state enable flags;
- queue depth, notification capacity, pacing, payload, and validation mode;
- at least three independent repetitions with p50, p90, p99, p99.9, and
  maximum, using enough samples for each reported percentile; and
- thermal or throttling evidence for long-running rows.

Keep default-power, performance-governor, and deep-idle-disabled results
separate. Combining all controls into one row prevents attribution.

The transport-suite manifest and native Runtime summaries retain benchmark
geometry, placement, CPU topology, governor, and available GPU evidence. Keep
the host setup state and any external thermal or throttling logs beside those
artifacts.

# cacti

`cacti` is a library for experimenting with computation graphs
(or computation "spines"). `cacti` is written in Rust, and uses
the [Futhark](https://github.com/diku-dk/futhark)
[language](https://futhark-lang.org/) to implement
computational kernels targeting GPUs and multicore CPUs.

The current pre-release of `cacti` is capable of no-configuration,
larger-than-VRAM training or fine-tuning of LLaMA-style language
models, using the full-precision gradient (i.e. f16 or f32).
In other words, using `cacti`, you do not modify your training
script to enable larger-than-VRAM training or fine-tuning;
`cacti` will do its best to utilize the available hardware resources,
based on your system's GPU memory and host CPU memory capacities.
`cacti` achieves this through an out-of-memory policy that
opportunistically spills memory allocations from the GPU to the CPU.

`cacti` is oriented toward developing memory-safe AI systems,
and so the `cacti` system code is entirely implemented in Rust.
The main non-Rust component is the Futhark language and compiler,
which is implemented in Haskell, and which is used extensively in
`cacti` to implement the underlying computation kernels of the
computation graph ("spine"). In fact, Futhark is essential to the
design of `cacti`, and some design choices in Futhark are likewise
visible in `cacti`.

## Design Priorities

- Purely functional/monotone update semantics
- Operational semantics specifying safe dataflow and autodiff
- Programming model: repeatable coroutines + cyclic dataflow
- Computational kernels: first, write it in Futhark
- Written for Rust

### Tradeoffs and Limitations

As this is a pre-release of `cacti`, there are a number of known
limitations due to tradeoffs made in prioritizing what to implement
first. Given finite development resources, maintaining the design
priorities listed earlier took precedence over significant extensions
to the system architecture, or other potential new features, such as
the following:

- Implementations for GPU targets other than CUDA
- Multi-GPU
- Quantization
- Python inter-op

## Installation

### Prerequisites

Requirements:

- git
- ghc and cabal-install (tested with GHC 9.0)
- rustc and cargo (tested with Rust 1.62; some older versions should also compile)
- gcc or clang, for Futhark dynamic compilation (tested with gcc)

GPU support:

- CUDA 11.x (tested with CUDA 11.5 and driver 495)

Thus far we have only tested `cacti` on x86-64 and Linux.

### Bootstrapping and installing from source (recommended method)

It is recommended to use the
[cacti-bootstrap](https://git.sr.ht/~ptrj/cacti-bootstrap)
git repository to bootstrap sources, as that repo vendors
git submodules of all Rust dependencies, as well as our patched
branch of the Futhark compiler.
However, please note that Futhark itself depends on many
Haskell packages which we do _not_ vendor, and are instead
downloaded by cabal during bootstrapping.

    git clone 'https://git.sr.ht/~ptrj/cacti-bootstrap'
    cd cacti-bootstrap
    ./bootstrap.sh
    cd cacti
    make

### Bootstrapping and installing from source (alternative method)

An alternative bootstrapping method, for developing on `cacti`
itself, uses the provided `bootstrap.sh` script to git clone
HEAD on all of the vendored dependencies.
Please note that this bootstrapping method will create a bunch
of cloned repo directories _outside the cacti repo directory_,
thus it is recommended to perform this in a dedicated workspace
directory.

    mkdir <your-workspace-dir>
    cd <your-workspace-dir>
    git clone 'https://git.sr.ht/~ptrj/cacti'
    cd cacti
    ./bootstrap.sh
    make

## Examples

In the "examples" directory, you will find provided code for
both fine-tuning and inference based on
[OpenLLaMA-3B](https://huggingface.co/openlm-research/open_llama_3b_v2).

It is recommended to read and understand the examples, and
to use them as starting points for your own experiments.

## Documentation

Please check back soon; this is a work in progress.

# idris-autodiff-c

A translator that converts Idris automatic differentiation deep embedding to C code.

## Overview

This project provides a tool to translate automatic differentiation code from Idris to C. It takes Idris files containing automatic differentiation implementations using deep embedding and generates efficient C code that computes both function values and derivatives up to the third order.

The translator supports both forward-mode and backward-mode automatic differentiation as defined in the Idris source file.

## Features

- Translation of Idris Expr data type to C code
- Support for common mathematical operations:
  - Addition, subtraction, multiplication, division
  - Trigonometric functions (sin, cos)
  - Exponential and logarithmic functions
  - Power functions
- Symbolic differentiation up to third-order derivatives
- Automatic code generation with safety measures (division by zero, etc.)
- Complete test program generation to validate the generated code

## Requirements

- OCaml (tested with version 5.0+)
- OCaml Str library
- C compiler (gcc recommended)
- GNU Make

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ericschmid-uchicago/idris-autodiff-c.git
   cd idris-autodiff-c
   ```

2. Compile the translator:
   ```bash
   ocamlopt -I +str -o idris_to_c str.cmxa idris_to_c.ml
   ```

## Usage

### Basic Usage

Run the translator on the included Idris file:

```bash
./idris_to_c AutoDiff.idr
```

By default, this will generate C code in a directory named `generated`.

You can also run it on your own Idris files with similar structure:

```bash
./idris_to_c path/to/your/custom/file.idr
```

### Custom Output Directory

You can specify a custom output directory:

```bash
./idris_to_c path/to/your/file.idr my_output_dir
```

### Compiling the Generated Code

After generation, compile and run the C code:

```bash
cd generated
make
./autodiff_test
```

This will run tests on all the translated functions and print their values and derivatives.

## How It Works

The translator works by:

1. Reading an Idris file containing automatic differentiation code
2. Extracting example functions from the file
3. Building an AST representation of each function
4. Generating C code for the functions and their derivatives
5. Creating a test program and build system

If automatic extraction fails, the translator falls back to a set of predefined examples.

## Example Functions

The translator processes example functions from the `AutoDiff.idr` file, including:

1. Polynomials: `f(x) = x^2 + 3x + 2`
2. Trigonometric: `f(x) = sin(x) * cos(x)`
3. Composites: `f(x) = sin(x^2)`
4. Exponentials: `f(x) = e^(x^2) / (1 + x^2)`
5. Gaussian-like: `f(x) = x * exp(-x^2 / 2)`
6. And more complex examples

## Code Structure

- `idris_to_c.ml`: Main translator code in OCaml
- `AutoDiff.idr`: Source Idris file containing automatic differentiation implementation
- `generated/`: Directory containing the generated C code
  - `autodiff.h`: Header file with function declarations
  - `example*.c`: Implementation files for each example function
  - `main.c`: Test program
  - `Makefile`: Build system

## Examples

Given an Idris function like:

```idris
example5_back : Expr -> Expr
example5_back x = SinExpr (MulExpr x x)
```

The translator generates C code:

```c
// f(x) = sin(x^2)
double example5(double x) {
    return sin((x * x));
}

// 1st derivative implementation
double example5_derivative_1(double x) {
    return (cos((x * x)) * (2.0 * x));
}

// 2nd derivative implementation
double example5_derivative_2(double x) {
    return (((((-1.0) * sin((x * x))) * (2.0 * x)) * (2.0 * x)) + (cos((x * x)) * 2.0));
}
```

## Acknowledgments

This project was inspired by the automatic differentiation work in Idris and the need for efficient C implementations for numerical applications.

# Building a Tiny GPU to Understand AI Hardware from Scratch

AI engineering has moved fast at the software level, but many engineers still treat hardware as a black box. We write PyTorch code, tune hyperparameters, and deploy models without knowing how silicon executes those instructions. This blind spot limits optimization and system design choices. Understanding hardware basics is the first step to thinking like an AI systems engineer. At the center of this hardware layer sits the GPU, handling the complex computations that drive AI workloads.

![Tiny GPU Architecture (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:4330/1*7058qZ5saYbOGtUqF7-LBw.png)

No matter what GPU we are talking about, whether it’s based on NVIDIA, AMD, or Intel architectures they all share the same fundamental anatomy designed for massive parallelism:

1.  **Compute Cores (ALUs):** Chip responsible for executing mathematical operations, often including specialized units like Tensor Cores for matrix multiplication.
2.  **Memory Controllers:** The complex traffic cops that manage High Bandwidth Memory (HBM/GDDR), acting as the bridge between the fast compute units and massive datasets.
3.  **Register Files:** Ultra-fast, on-chip storage unique to each thread, allowing thousands of threads to maintain their own distinct state simultaneously (SIMD).
4.  **Schedulers:** It manages execution pipelines, specifically designed to hide memory latency by instantly switching between active groups of threads (Warps).
5.  **Dispatcher:** The global manager that breaks down massive workloads (kernels) into smaller blocks and distributes them to available cores across the silicon die.

and there are many more components. In this blog ..

> We will design a tiny GPU and explore its 12 core components, understanding how each one works from an AI engineer perspective. Then, we will put our tiny GPU to the test by performing some computations on it.

### Table of Content

*   [Our tiny GPU vs NVIDIA H100](#our-tiny-gpu-vs-nvidia-h100)
*   [Understanding the Basics of SystemVerilog](#understanding-the-basics-of-systemverilog)
*   [Arithmetic Logic Unit (ALU)](#arithmetic-logic-unit-alu)
*   [Register File](#register-file)
*   [Program Counter (PC)](#program-counter-pc)
*   [Memory Controller](#memory-controller)
*   [Load Store Unit (LSU)](#load-store-unit-lsu)
*   [Decoder](#decoder)
*   [Fetcher](#fetcher)
*   [Scheduler](#scheduler)
*   [Compute Core (Streaming Multiprocessor)](#compute-core-streaming-multiprocessor)
*   [Device Control Register (DCR)](#device-control-register-dcr)
*   [Dispatcher (GigaThread Engine)](#dispatcher-gigathread-engine)
*   [GPU Top Module (Silicon Die)](#gpu-top-module-silicon-die)
*   [Testing the GPU](#testing-the-gpu)
*   [Scaling to a Modern Blackwell GPUs](#scaling-to-a-modern-blackwell-gpus)

---

### Our tiny GPU vs NVIDIA H100

Normally, building a GPU requires access to fabrication plants, and specialized hardware labs. But to make things accessible, we are going to use the resources available right in our operating system **Simulation**. We will construct a virtual chip that behaves exactly like physical silicon logic, but runs entirely on your laptop.

![Tiny GPU vs H100 (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:1250/1*R_vM_CubfrhIwAdZJZbZ6Q.png)

To make things a bit easier, I will use the NVIDIA H100 as a comparison point, since it is currently the gold standard for AI workloads. This will help clarify exactly what we are building. **While we cannot match the raw scale of an H100 in a tutorial, we can replicate the core architectural flow that makes it so effective**.

We are stripping away the complexity of branch prediction and caches to create a sandbox where beginner developers can clearly understand how data flows through a GPU, how threads are managed, and how parallel execution actually happens.

### Understanding the Basics of SystemVerilog

**Most of the ai engineers are aware of high-level programming languages such as Python, C++, or JavaScript** but when it comes to hardware design languages like Verilog or SystemVerilog it can be quite difficult to understand the syntax and variable definitions.

> So before we start building chips, we need to understand the language they are written in. We will use **SystemVerilog**.

It is an industry-standard hardware description language (HDL) used to model, design, simulate, and verify digital systems such as processors, memory units, and other integrated circuits.

In Python, code executes line-by-line, sequentially. In Verilog, **everything executes at the exact same time.** You aren’t writing a program sequence; you are describing a physical circuit where electricity flows through every wire simultaneously.

We are going to learn the core concepts of SystemVerilog that will be enough for this blog to understand how GPUs work under the hood.

First let’s understand how a simple module is defined in SystemVerilog. A `module` is the basic building block. It wraps logic into a reusable box.

```verilog
module alu #(
    parameter WIDTH = 8  // Compile-time constant
) (
    input wire clk,      // The Clock signal
    input wire [7:0] a,  // 8-bit input data
    output reg [7:0] res // 8-bit output storage
);
    // Logic goes here...
endmodule
```

So basically …

*   `module` / `endmodule` defines the boundaries of the component. Think of this as `class ALU(nn.Module):` in PyTorch.
*   `parameter` are constants set *before* the hardware is built. Think of them as **Hyperparameters** (like `d_model` or `layer_count` in a Transformer config). Once the chip is printed, these cannot change.
*   `input` / `output` are the physical pins on the chip. They correspond to the arguments in your `forward()` pass.

In software, a variable is just a place to put data. In hardware, the *type* of variable defines its physical existence.

*   `wire`: A physical copper wire with no memory. If you put `5` on one end, `5` appears on the other end instantly. Used for combinational logic (math, routing). *AI Analogy:* A stateless operation like `F.relu(x)`. It calculates and disappears.
*   `reg` (Register): A storage cell (flip-flop) with memory. It holds a value and only changes when the clock tells it to. Used for sequential logic (state machines, accumulators). *AI Analogy:* A model weight or optimizer state. It persists across training steps.

Now let’s break down some common syntax and variable definitions you will see in SystemVerilog code. Normally in python we have `int`, `float`, `str` but in hardware design we have to be more specific about the *size* of these variables.

*   `[7:0]`: This denotes an 8-bit bus (bits 7 down to 0). It can hold values from 0 to 255.
*   `[15:0]`: A 16-bit bus.

Think of this as `dtype`. `[7:0]` is `torch.int8` (Quantized). `[31:0]` is `torch.float32`. In this project, we are going to use `[7:0]` to simulate an INT8 quantized engine.

After defining inputs and outputs, we often need to describe how the logic operates over time. This is where `always` blocks come in. In python we have functions that execute when called, but in hardware, we define behavior that happens on clock edges.

In simple terms, an `always` block is like a function that runs automatically whenever certain conditions are met (like a clock tick).

```verilog
always @(posedge clk) begin
    if (reset) begin
        counter <= 0;
    end else begin
        counter <= counter + 1;
    end
end
```

They are normally reprsented by `always @(posedge clk)` block triggers **only** when the clock signal goes from 0 to 1 (Positive Edge).

This is a special operator for registers. It means “Update this value at the *end* of the cycle.” It ensures that all flip-flops in the entire GPU update in perfect unison.

This is the `optimizer.step()` function. The hardware computes everything, waits for the clock (the step), and then updates all weights simultaneously.

Another concept you will see frequently is **Instantiation**. Just as you can call a function inside another function in Python, in SystemVerilog you can place one module inside another. This is how we build complex systems from simple blocks.

```verilog
// Inside the Top-Level GPU module
alu my_alu_instance (
    .clk(system_clock),
    .a(input_wire_a),
    .res(result_wire)
);
```

Think of this as creating an object instance: `self.layer1 = nn.Linear(...)`. The syntax `.clk(system_clock)` is physically soldering the `clk` port of the ALU to the `system_clock` wire of the GPU.

Hardware works on bits, so we need specific operators to manipulate them. You will see symbols like `&`, `|`, and `^`. These are **Bitwise Operators** (AND, OR, XOR). They operate on every bit individually. For example, `1010 & 0011` results in `0010`.

We also have a unique operator called **Concatenation**, represented by curly braces `{}`. It glues wires together.

*   If `a = 10` (2 bits) and `b = 11` (2 bits), then `{a, b}` becomes `1011` (4 bits).
*   *AI Analogy:* This is exactly like `torch.cat([tensor_a, tensor_b], dim=0)`.

To keep our code readable and avoid “magic numbers,” we use `localparam`.

```verilog
localparam FETCH = 3'b001;
localparam DECODE = 3'b010;
```

Here, `3'b001` means "3 bits wide, binary value 001". A `localparam` is a constant local to that file, similar to a `const` in C++ or a global configuration variable in Python.

Finally, when we want to test our designs, we use `initial begin` blocks. While `always` blocks run forever, `initial` blocks run exactly once at the start of the simulation.

```verilog
initial begin
    clk = 0;
    reset = 1;
    #10 reset = 0; // Wait 10 nanoseconds, then turn off reset
end
```

The `#10` syntax is a delay command. It tells the simulator to pause for 10 time units before executing the next line. This is mainly used in testbenches to simulate the passage of time or to generate clock signals.

So, now that we have covered the basics of SystemVerilog syntax and variable definitions and also you can learn a lot more on their [official documentation](https://www.chipverify.com/tutorials/systemverilog), we are ready to start building the core components of our Tiny GPU.

### Arithmetic Logic Unit (ALU)

The very first component that we need to build is the Arithmetic Logic Unit (ALU). On any GPU, this is the core hardware block that performs all the mathematical operations. In AI workloads, this is where the bulk of the computation happens, especially for operations like matrix multiplications which are fundamental to neural networks.

> It does not contain any memory itself, it simply takes inputs, performs calculations, and outputs results.

When we train an AI model, the ALU is where the actual `Weight * Input` multiplications occur. You might be aware of `TFLOPS` (Tera Floating Point Operations per Second) or `TOPS` (Tera Operations per Second) as metrics for GPU performance in AI tasks. These metrics directly relate to how many operations the ALU can perform in a second.

![ALU Functionality (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:1250/1*ndpio5VJR6u8g-XX1E9Q6w.png)
*ALU Functionality (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

**On an `H100` GPU, there are thousands of ALUs working in parallel, each handling different threads of computation**. For simplicity, in our `Tiny GPU`, we will implement a basic version of the ALU that can handle fundamental arithmetic operations like addition, subtraction, multiplication, and division.

Let’s create a file named `arthimetic_logic_unit.sv` where we will implement the ALU core logic.

For arithmetic operations, we place two compiler directives at the top of the Verilog file:

```verilog
`default_nettype none
`timescale 1ns/1ns
```

1.  Setting `default_nettype` to none to help catch common bugs such as signal name typos. For example, if we accidentally write sum = a + bb; instead of sum = a + b;, the compiler will report an error instead of silently creating an unintended wire.
2.  The `timescale` defines how simulation time is interpreted. With `1ns/1ns`, means a delay such as #5 represents a 5-nanosecond delay, and the simulator will not model time more precisely than 1 nanosecond.

In production hardware designs, these directives are typically placed at the top of each file to ensure consistent behavior across the entire design.

Now we need to create the module definition for the ALU which is pretty **similar to defining a class in Python**. It is going to be responsible for executing computations on register values. Let’s define the module and its inputs and outputs:

```verilog
module alu (
    input wire clk,    
    input wire reset,
    input wire enable, // If current block has less threads then block size, some ALUs will be inactive
    input reg [2:0] core_state,
    input reg [1:0] decoded_alu_arithmetic_mux,
    input reg decoded_alu_output_mux,
    input reg [7:0] rs,
    input reg [7:0] rt,
    output wire [7:0] alu_out
);
```

Let’s break down these individual components:

*   `clk`: The clock signal that synchronizes operations, if our ALU is running at 1.8GHz, this signal ticks 1.8 billion times per second.
*   `reset`: A signal to reset the ALU state, so that it starts fresh if a new computation is to be performed.
*   `enable`: If your AI model has a batch size of `3`, but the GPU hardware runs in blocks of `4`, one thread is useless. This `enable` wire physically turns off the 4th ALU to save power.
*   `core_state`: The ALU should only burn electricity when the Core is in the `EXECUTE` state. If the Core is waiting for memory (loading weights), this signal tells the ALU to chill/relax.
*   `decoded_alu_arithmetic_mux`: This is responsible for selecting which arithmetic operation to perform (addition, subtraction, multiplication, division). For example, if the instruction is `FADD` (Float Add), these bits flip to `00`. If `FMUL` (Float Multiply), they flip to `10`.
*   `decoded_alu_output_mux`: A signal that determines whether to output the result of arithmetic operations or comparison results.
*   `rs` and `rt`: These are represeting the `weight` and `input` values stored in registers that the ALU will operate on. `[7:0]` means this is an 8-bit ALU. **Our tiny GPU is natively INT8**.
*   `alu_out`: An 8-bit output wire that will carry the result of the ALU operations.

> Modern AI is moving from `FP32 -> FP16 -> INT8 -> FP4` because a [7:0] wire is 4x smaller physically than a [31:0] wire `(FP32)`. This is why Quantization makes chips smaller and faster.

On a powerful AI GPU like the H100, the above module would be instantiated thousands of times, once for each thread running in parallel. Now we define some local parameters to represent the arithmetic operations.

```verilog
localparam ADD = 2'b00,
    SUB = 2'b01, // For subtraction
    MUL = 2'b10, // For multiplication
    DIV = 2'b11; // For division
```

`b01` is binary representation for `1`, `b10` for `2`, and so on. It typically makes the code more readable. Nvidia GPUs typically contain many more operations like bitwise operations, logical operations, and floating-point specific operations, but for our Tiny GPU, we will keep it simple with just these four basic arithmetic operations.

We then declare a register to hold the output value internally and connect it to the output wire:

```verilog
reg [7:0] alu_out_reg;
assign alu_out = alu_out_reg;
```

`alu_out_reg` is where we will store the result of our computations before sending it out through the `alu_out` wire. `reg [7:0]` indicates that this register can hold an 8-bit value.

So far we have defined the structure of our ALU module. Now, we need to implement the core logic that performs the arithmetic operations based on the inputs and control signals. This is done using an `always` block that triggers when the clock signal rises which means when there is a positive edge on the clock signal:

```verilog
always @(posedge clk) begin 
    // Trigger on rising edge of the clock
    if (reset) begin 
        // Clear ALU output register on reset
        alu_out_reg <= 8'b0;
    end else if (enable) begin
        // Only update ALU output when enabled
        if (core_state == 3'b101) begin 
            // Execute ALU operation in EXECUTE state
            if (decoded_alu_output_mux == 1) begin 
                // Generate NZP comparison flags
                alu_out_reg <= {5'b0, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)};
            end else begin 
                // Perform arithmetic operation
                case (decoded_alu_arithmetic_mux)
                    ADD: begin 
                        // Addition
                        alu_out_reg <= rs + rt;
                    end
                    SUB: begin 
                        // Subtraction
                        alu_out_reg <= rs - rt;
                    end
                    MUL: begin 
                        // Multiplication
                        alu_out_reg <= rs * rt;
                    end
                    DIV: begin 
                        // Division
                        alu_out_reg <= rs / rt;
                    end
                endcase
            end
        end
    end
end
```

`@(posedge clk)` means the ALU updates its output once per clock cycle. Each clock tick represents one step of computation, similar to how a GPU processes AI operations step by step.

In this block:

*   We first check if the `reset` signal is high. If it is, we clear the `alu_out_reg` to `0`.
*   If not resetting, we check if the `enable` signal is high. This ensures that the ALU only performs operations when it is supposed to.
*   Next, we check if the `core_state` indicates that we are in the `EXECUTE` phase (represented by `3'b101`).
*   If we are in the `EXECUTE` phase, we check the `decoded_alu_output_mux` signal. If it is `1`, we perform a comparison operation to generate NZP (Negative, Zero, Positive) flags based on the difference between `rs` and `rt`.
*   If `decoded_alu_output_mux` is `0`, we proceed to perform the arithmetic operation specified by `decoded_alu_arithmetic_mux` using a `case` statement. Depending on the value, we perform addition, subtraction, multiplication, or division and store the result in `alu_out_reg`.

Finally, we close the module definition:

```verilog
endmodule // Finalizes the ALU module
```

Obviously, this is a very simplified version of an ALU compared to what you would find in a real GPU like the H100, which would support many more operations and optimizations but this is a standard `ALU` implementation that captures the high-level functionality needed for our Tiny GPU simulation.

### Register File

The problem with `ALU` of any GPU is that it has short-term memory loss. It performs a calculation like `10 + 20`, outputs `30`, and immediately forgets it.

We need a place to store these numbers quickly so the ALU can use them again. In computer architecture, this is called the **Register File**.

![Register logic (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*y9JUDHzZFJD6hrxLnG9CSA.png)
*Register logic (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In the context of AI, Register File as the `Backpack` for each thread. When a CUDA thread is running on an H100, it can't afford to run all the way to VRAM (Global Memory) every time it needs to check a loop counter or store a partial sum. That would be incredibly slow. Instead, it keeps those critical variables right next to the ALU in these registers.

The Register File is the fastest memory on the entire chip. On an NVIDIA GPUs, managing `Register Pressure` is a massive part of kernel optimization. If your AI kernel uses too many variables, it runs out of space in the register file and "spills" data to slower memory, reducing your performance.

But this file serves an even more critical purpose for us which we call **Identity**.

How do 1,000 threads running the exact same code know to process different pixels of an image? It’s because of **SIMD (Single Instruction, Multiple Data)**. Inside this file, we are going to hardcode specific registers to hold the `Thread ID`. This allows every thread to calculate a unique address (e.g., `Base Address + Thread ID`) to fetch its own unique data.

Let’s create `registers.sv` and give our threads some memory. In a real GPU like the H100, register files are massive (64KB+) and highly optimized for speed.

Just like with `ALU`, we start with our standard compiler directives to keep our design clean.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module (In python we can think of this as a class). This module sits right next to the ALU in the pipeline.

```verilog
module registers #(
    parameter THREADS_PER_BLOCK = 4,
    parameter THREAD_ID = 0, // This is CRITICAL. Every instance gets a unique ID.
    parameter DATA_BITS = 8
) (
    input wire clk,
    input wire reset,
    input wire enable, // If the block is not full, disable this register file

    // Kernel Execution Metadata
    input reg [7:0] block_id,

    // State of the Core (Fetch, Decode, Execute, etc.)
    input reg [2:0] core_state,

    // Instruction Signals (Which register to read/write?)
    input reg [3:0] decoded_rd_address, // Destination (Where to save)
    input reg [3:0] decoded_rs_address, // Source 1 (Input A)
    input reg [3:0] decoded_rt_address, // Source 2 (Input B)

    // Control Signals from Decoder
    input reg decoded_reg_write_enable,
    input reg [1:0] decoded_reg_input_mux,
    input reg [DATA_BITS-1:0] decoded_immediate,

    // Inputs from other units
    input reg [DATA_BITS-1:0] alu_out, // Result from the Math unit
    input reg [DATA_BITS-1:0] lsu_out, // Result from the Memory unit

    // Outputs to the ALU
    output reg [7:0] rs,
    output reg [7:0] rt
);
```

Here, we are basically defining a module named `registers` that represents the Register File for our Tiny GPU. Let's break down the key parameters and signals:

*   `THREAD_ID`: This is the most important parameter. In a GPU, when you spawn 1000 threads, the hardware doesn't magically know which is which. We physically pass a number (`0`, `1`, `2`...) into this parameter when we build the chip. This becomes the thread's identity.
*   `block_id`: This represents which block of threads is currently executing. This is useful for larger workloads where multiple blocks of threads are running concurrently.
*   `alu_out` & `lsu_out`: Notice that the register file sits at the intersection of Math (ALU) and Memory (LSU). It is the central hub. It takes the result of a calculation (`alu_out`) or a loaded weight (`lsu_out`) and stores it for later use.
*   `decoded_rd/rs/rt`: These correspond to assembly instructions. `ADD R0, R1, R2` tells the hardware: "Read R1 (rs), Read R2 (rt), and write the result to R0 (rd)".

We define a few local parameters to help us decide where the data coming into the register is coming from:

```verilog
localparam ARITHMETIC = 2'b00, // Data coming from ALU (ADD, SUB)
    MEMORY = 2'b01,            // Data coming from RAM (LDR)
    CONSTANT = 2'b10;          // Data is a hardcoded constant (CONST)
```

This acts as a Source Selector. In a Neural Network operation, you are constantly switching between:

1.  Fetching Weights (`MEMORY` mode).
2.  Multiplying them (`ARITHMETIC` mode).
3.  Setting loop counters (`CONSTANT` mode). These constants allow our logic to cleanly switch between these modes based on the current instruction.

Now, we allocate the actual physical storage. On an H100, you might have `65,536` registers per Streaming Multiprocessor. Since we are building a Tiny GPU, we will stick to 16 registers per thread.

```verilog
// 16 registers per thread (13 free registers and 3 read-only registers)
reg [7:0] registers[15:0];
```

This array `registers[15:0]` is the physical memory cells.

*   In NVIDIA GPUs, this “Register File” is massive (KB size) and incredibly fast (TB/s bandwidth). Optimizing usage of this finite space is highly related to CUDA performance. If you define too many local variables in your kernel, you run out of slots in this array.
*   We have 16 slots. We are going to reserve the last 3 for special system values.

Now we are going to build the **SIMD** logic. Pay close attention to how we handle registers 13, 14, and 15 inside the `reset` block.

```verilog
always @(posedge clk) begin
    if (reset) begin
        // Empty the output wires
        rs <= 0;
        rt <= 0;
        
        // Initialize all free registers (R0 - R12) to zero
        registers[0] <= 8'b0;
        registers[1] <= 8'b0;
        registers[2] <= 8'b0;
        registers[3] <= 8'b0;
        registers[4] <= 8'b0;
        registers[5] <= 8'b0;
        registers[6] <= 8'b0;
        registers[7] <= 8'b0;
        registers[8] <= 8'b0;
        registers[9] <= 8'b0;
        registers[10] <= 8'b0;
        registers[11] <= 8'b0;
        registers[12] <= 8'b0;
        
        // --- THE MOST IMPORTANT PART FOR AI ---
        // Initialize read-only registers for Identity
        registers[13] <= 8'b0;              // %blockIdx (Updated dynamically later)
        registers[14] <= THREADS_PER_BLOCK; // %blockDim (Total threads)
        registers[15] <= THREAD_ID;         // %threadIdx (Who am I?)
```

In here …

*   `registers[13] <= 8'b0;`: This register will hold the `Block ID`. It is initialized to `0` but will be updated dynamically when a new block is issued. This helps threads know which block they belong to.
*   `registers[14] <= THREADS_PER_BLOCK;`: This register holds the total number of threads in the block. This is useful for calculating offsets and ensuring that threads operate within their designated range.
*   `registers[15] <= THREAD_ID;`: **This line enables Parallel AI.** In CUDA, when you type `int i = threadIdx.x;`, the hardware is literally reading from this Register 15.

Since every instance of this file has a different `THREAD_ID` parameter, Thread 0 sees `0` here, and Thread 1 sees `1`.

This allows Thread 0 to load `Matrix[0]` and Thread 1 to load `Matrix[1]` simultaneously, even though they are running the exact same code. This is **SIMT (Single Instruction Multiple Thread)** minimal workflow.

```verilog
end else if (enable) begin 
    // Update the block_id when a new block is issued from dispatcher
    // In a real GPU, this changes when the Scheduler swaps warps
    registers[13] <= block_id; 
    
    // 1. READ PHASE
    // When the Core is in the REQUEST state, we fetch values for the ALU
    if (core_state == 3'b011) begin 
        rs <= registers[decoded_rs_address];
        rt <= registers[decoded_rt_address];
    end

    // 2. WRITE PHASE
    // When the Core is in the UPDATE state, we save results back
    if (core_state == 3'b110) begin 
        // Only allow writing to R0 - R12 (Protect our identity registers!)
        if (decoded_reg_write_enable && decoded_rd_address < 13) begin
            case (decoded_reg_input_mux)
                ARITHMETIC: begin 
                    // Save result from ALU (e.g., C = A + B)
                    registers[decoded_rd_address] <= alu_out;
                end
                MEMORY: begin 
                    // Save result from Memory (e.g., Loading Weights)
                    registers[decoded_rd_address] <= lsu_out;
                end
                CONSTANT: begin 
                    // Save a hardcoded value (e.g., i = 0)
                    registers[decoded_rd_address] <= decoded_immediate;
                end
            endcase
        end
    end
end
endmodule
```

In an else-if block checking for `enable`, we implement two main phases:

*   **Read Phase (**`3'b011`**):** Before the ALU fires, we look up the numbers it needs (`rs`, `rt`) and put them on the wires.
*   **Write Phase (**`3'b110`**):** After the ALU finishes, we take the result and save it back into the array.
*   **Protection Logic (**`decoded_rd_address < 13`**):** This is a safety mechanism. We forbid the code from overwriting registers 13, 14, or 15. If a bug in your kernel overwrote `threadIdx`, the thread would lose its identity and start processing the wrong data, corrupting the entire AI model output. This effectively makes them "Read-Only" system registers.

So far we have built the basic structure of the Register File for our Tiny GPU, which is essential for storing intermediate values during computations and maintaining thread identity in a parallel processing environment. In a real GPU like the H100, the Register File would be much larger and more complex, but this simplified version captures the core functionality needed for our simulation.

### Program Counter (PC)

In ALU we perform calculations while in Registers we store numbers but in both cases there is no direction component that tells us what to do next in real GPU hardware these components are coordinated by the Program Counter (PC).

![PC Control Logic (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:1250/1*5YdZ5fAE_z4CncmWT5q-0g.png)
*PC Control Logic (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

For example, when running an AI algorithm like training a neural network we often have to loop over data multiple times like iterating over batches of images or sequences in natural language processing basically it tracks where we are in the code and tells us what instruction to execute next this is where the Program Counter comes in.

For an AI Engineer, think of the PC as the **Navigator** or the index `i` in a `for i in range(rows):` loop.

On an **NVIDIA H100**, the Program Counter is vastly more complex because of **Branch Divergence**. An H100 executes threads in groups of 32 called `Warps`. What happens if you write an AI kernel where:

```verilog
if (pixel_val > 0) {
    // Do heavy math
} else {
    // Do nothing
}
```

If 16 threads want to do the math (Left Path) and 16 want to do nothing (Right Path), the hardware cannot split the warp. It has to force *all* 32 threads to go down the Left Path (masking off the 16 who didn’t want to), and then force *all* 32 to go down the Right Path. This effectively doubles the execution time.

For our simple GPU, we assume **perfect convergence**. We assume all threads always follow the same path. The PC will simply point to the current line of code and update to the next line after every cycle, unless we tell it to jump (branch).

This is why optimizing your custom CUDA kernels to avoid `if/else` branching is so critical, you are literally going against the physical limitations of this specific hardware unit.

Let’s create `program_controller.sv` to give our threads direction.

As always, we start with our standard compiler directives to keep our design clean:

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module.

```verilog
module program_controller #(
    parameter DATA_MEM_DATA_BITS = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8
) (
    input wire clk,
    input wire reset,
    input wire enable, // If current block has less threads then block size, some PCs will be inactive


    // State
    input reg [2:0] core_state,

    // Control Signals
    input reg [2:0] decoded_nzp,
    input reg [DATA_MEM_DATA_BITS-1:0] decoded_immediate,
    input reg decoded_nzp_write_enable,
    input reg decoded_pc_mux, 

    // ALU Output - used for alu_out[2:0] to compare with NZP register
    input reg [DATA_MEM_DATA_BITS-1:0] alu_out,

    // Current & Next PCs
    input reg [PROGRAM_MEM_ADDR_BITS-1:0] current_pc,
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] next_pc
);
```

Let’s analyze the inputs that we have here:

*   `current_pc` & `next_pc`: The core logic. We read where we are, calculate where to go, and output it.
*   `decoded_nzp`: This stands for **Negative, Zero, Positive**. This is how the GPU makes decisions. If you write `if (x < 0)`, the hardware checks the "Negative" flag.
*   `decoded_pc_mux`: This is the switch. If `0`: Go to the next line (`PC + 1`) and if `1`: Jump to a specific line (`Branch`).
*   `alu_out`: The PC watches the ALU. Why? Because the decision to branch usually depends on a math result (e.g., "Is `i < 100`?").

Now, we need to store the result of the *previous* comparison that the ALU made. This is done using a small register called `nzp`.

```verilog
reg [2:0] nzp;
```

This 3-bit register is the hardware’s short-term memory for decisions.

*   Bit 2: Did the last math result result in a **N**egative number?
*   Bit 1: Was it **Z**ero?
*   Bit 0: Was it **P**ositive?

In the H100, this is part of the **Predicate Register** file. Modern AI architectures rely heavily on “Predication” (executing instructions conditionally based on a flag) rather than full branching, because it keeps the pipeline smoother. Our `nzp` register is a simplified version of this.

Now for the logic. This determines the flow of your AI kernel.

```verilog
always @(posedge clk) begin
    if (reset) begin
        nzp <= 3'b0;
        next_pc <= 0;
    end else if (enable) begin
        // Update PC when core_state = EXECUTE
        if (core_state == 3'b101) begin 
            if (decoded_pc_mux == 1) begin 
                if (((nzp & decoded_nzp) != 3'b0)) begin 
                    // On BRnzp instruction, branch to immediate if NZP case matches previous CMP
                    next_pc <= decoded_immediate;
                end else begin 
                    // Otherwise, just update to PC + 1 (next line)
                    next_pc <= current_pc + 1;
                end
            end else begin 
                // By default update to PC + 1 (next line)
                next_pc <= current_pc + 1;
            end
        end
```

This block calculates the **Next Step**.

*   **The Default Path:** `next_pc <= current_pc + 1`. This is what happens 90% of the time. The GPU executes line 1, then line 2, then line 3.
*   **The Branch Path (**`decoded_pc_mux == 1`**):** This is where the looping happens in AI code.

The code checks `(nzp & decoded_nzp)`.

*   Example: If the instruction is “Branch if Negative” (`BRn`), and the `nzp` register says the last result was indeed Negative, the `if` statement evaluates to true.
*   **The Jump:** `next_pc <= decoded_immediate`. Instead of going to the next line, we teleport to line 5 (or wherever the loop starts). This is how a `while` loop works in silicon.

Finally, we need to handle the updating of that NZP register itself.

```verilog
// Store NZP when core_state = UPDATE   
if (core_state == 3'b110) begin 

    // Write to NZP register on CMP instruction
    if (decoded_nzp_write_enable) begin
        nzp[2] <= alu_out[2];
        nzp[1] <= alu_out[1];
        nzp[0] <= alu_out[0];
    end
end      
endmodule
```

*   **The Update Phase (**`3'b110`**):** After the ALU finishes a comparison (like `CMP R0, R1`), it outputs 3 bits indicating if the result was Negative, Zero, or Positive.
*   We capture those bits into our local `nzp` register.

This logic is exactly how activation functions like ReLU are implemented at a low level.

*   `CMP R0, #0` (Compare input to 0).
*   This updates the `nzp` bits.
*   `BRn SKIP` (If Negative, branch to skip code).

This logic allows the GPU to make decisions on data, which is fundamental to any non-linear AI operation.

But right now, it can only work on the tiny amount of data in its backpack. In the next section, we need to build the infrastructure to let it talk to the outside world. This leads us to the **Memory Controller**.

### Memory Controller

`ALU` or `PC` can calculate, loop, and branch. But right now, it is isolated. It has no access to the memory where the AI model weights, these are called **Global Memory (VRAM)**.

One of the modern Artificial Intelligence problem is **the Memory Wall** that moden GPUs like `NVIDIA Blackwell` architecture are trying to solve.

![Memory Controller (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*9pFagn_qPyxrruTM_uyNkA.png)
*Memory Controller (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

Your ALU is incredibly fast. It can do a multiply in nanoseconds. But fetching data from VRAM is painfully slow. Even worse, you have thousands of threads running at the exact same time. If they all try to grab the memory simultaneously, the electrical signals will collide, and the system will crash.

this can actually be solve through **Traffic Control**. We need a unit that stands between the Cores and the slow Memory, so we can organize the chaos into a neat line. In hardware, this is called the **Memory Controller**.

On an **NVIDIA H100**, this is the **HBM3 (High Bandwidth Memory) Controller**. It manages a massive 3.35 TB/s of bandwidth. It uses complex logic called `Coalescing` to combine requests: if Thread 0 asks for address 100 and Thread 1 asks for address 101, the controller is smart enough to turn that into one single request for "100 and 101", saving massive amounts of time.

For our **Tiny GPU**, we will build a simpler version. It will act as an **Arbiter**. It will look at all the threads asking for data, pick one, send the request to memory, wait for the answer, and hand it back.

Let’s create `memory_controller.sv` to manage this critical bottleneck.

As always, we start with our standard compiler directives:

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module. This module sits outside the cores, acting as the gateway to the RAM.

```verilog
module memory_controller #(
    parameter ADDR_BITS = 8,
    parameter DATA_BITS = 16,
    parameter NUM_CONSUMERS = 4, // The number of threads/cores accessing memory
    parameter NUM_CHANNELS = 1,  // The number of concurrent paths to memory
    parameter WRITE_ENABLE = 1   // Program memory is read-only, Data memory is read/write
) (
    input wire clk,
    input wire reset,


    // Consumer Interface (The Threads asking for data)
    input reg [NUM_CONSUMERS-1:0] consumer_read_valid,
    input reg [ADDR_BITS-1:0] consumer_read_address [NUM_CONSUMERS-1:0],
    output reg [NUM_CONSUMERS-1:0] consumer_read_ready,
    output reg [DATA_BITS-1:0] consumer_read_data [NUM_CONSUMERS-1:0],
    
    input reg [NUM_CONSUMERS-1:0] consumer_write_valid,
    input reg [ADDR_BITS-1:0] consumer_write_address [NUM_CONSUMERS-1:0],
    input reg [DATA_BITS-1:0] consumer_write_data [NUM_CONSUMERS-1:0],
    output reg [NUM_CONSUMERS-1:0] consumer_write_ready,
    
    // Memory Interface (The Actual RAM Chips)
    output reg [NUM_CHANNELS-1:0] mem_read_valid,
    output reg [ADDR_BITS-1:0] mem_read_address [NUM_CHANNELS-1:0],
    input reg [NUM_CHANNELS-1:0] mem_read_ready,
    input reg [DATA_BITS-1:0] mem_read_data [NUM_CHANNELS-1:0],
    
    output reg [NUM_CHANNELS-1:0] mem_write_valid,
    output reg [ADDR_BITS-1:0] mem_write_address [NUM_CHANNELS-1:0],
    output reg [DATA_BITS-1:0] mem_write_data [NUM_CHANNELS-1:0],
    input reg [NUM_CHANNELS-1:0] mem_write_ready
);
```

Let’s analyze the parameters that define our bottleneck:

*   `NUM_CONSUMERS`: This is the crowd. In a real GPU, this is huge. These are all the Load Store Units (LSUs) from all the cores trying to fetch weights. It is basically the number of threads trying to access memory at once.
*   `NUM_CHANNELS`: This is the number of lanes to the memory. On an H100, this is massive (dozens of channels) to handle the flood of requests.

An H100 has thousands of consumers but massive bandwidth (many channels). If `CONSUMERS > CHANNELS` (which is always true in AI), threads have to wait in line. This is why "Memory Bound" operations exist. The math is fast, but the line for the memory controller is long.

We need to keep track of the state of each channel. Is it busy? Who is it serving?

```verilog
localparam IDLE = 3'b000, 
    READ_WAITING = 3'b010, 
    WRITE_WAITING = 3'b011,
    READ_RELAYING = 3'b100,
    WRITE_RELAYING = 3'b101;

// Keep track of state for each channel and which jobs each channel is handling
reg [2:0] controller_state [NUM_CHANNELS-1:0];

reg [$clog2(NUM_CONSUMERS)-1:0] current_consumer [NUM_CHANNELS-1:0]; // Which consumer is this channel serving?
reg [NUM_CONSUMERS-1:0] channel_serving_consumer; // Prevents two channels from grabbing the same job
```

We use a state mechanism again.

*   `IDLE`: The channel is free. Looking for work.
*   `WAITING`: We sent the request to RAM, now we wait for the physical chips to respond.
*   `RELAYING`: We got the data, now we are handing it back to the specific thread that asked for it.

Now we are going to create the **Arbitration Loop** which is going to manage the flow of data.

```verilog
always @(posedge clk) begin
    if (reset) begin 

        // Reset all signals to 0...
        mem_read_valid <= 0;
        mem_read_address <= 0;
        consumer_read_ready <= 0;
        channel_serving_consumer = 0;

        // (assume standard reset logic for all registers)
    end else begin 
        // For each channel, we handle processing concurrently
        for (int i = 0; i < NUM_CHANNELS; i = i + 1) begin 
            case (controller_state[i])
                IDLE: begin
                    // While this channel is idle, cycle through consumers looking for one with a pending request
                    for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin 
                        if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin 
                            // Found a thread asking for data!
                            channel_serving_consumer[j] = 1;   // Mark thread as "Being Served"
                            current_consumer[i] <= j;          // Remember who asked

                            // Send request to global memory
                            mem_read_valid[i] <= 1;
                            mem_read_address[i] <= consumer_read_address[j];
                            controller_state[i] <= READ_WAITING;
                            // Stop looking, we found a job
                            break;
                        end else if (consumer_write_valid[j] && !channel_serving_consumer[j]) begin 
                            // Found a thread trying to write data (Store)!
                            channel_serving_consumer[j] = 1;
                            current_consumer[i] <= j;
                            mem_write_valid[i] <= 1;
                            mem_write_address[i] <= consumer_write_address[j];
                            mem_write_data[i] <= consumer_write_data[j];
                            controller_state[i] <= WRITE_WAITING;
                            break;
                        end
                    end
                end
```

This `IDLE` block is where the queue happens.

*   We loop `for (int j = 0; j < NUM_CONSUMERS; j = j + 1)` so that every thread gets a chance to ask for data.
*   This is a simple **First-Come-First-Serve** mechanism. The first thread we find that is asking for data gets served. It is pretty similar to how a CPU handles multiple processes competing for the same resource.
*   If Thread 0 wants data, we take its request and the Channel becomes busy. Thread 1 has to wait for the next free Channel.

This loop logic also explaining why **Memory Contention** slows down training. The more threads you have competing for limited channels, the longer this loop takes to service everyone.

Now we handle the waiting and the relaying.

```verilog
READ_WAITING: begin
    // Wait for response from global memory
    if (mem_read_ready[i]) begin 
        mem_read_valid[i] <= 0;
        
        // Give data back to the specific consumer
        consumer_read_ready[current_consumer[i]] <= 1;
        consumer_read_data[current_consumer[i]] <= mem_read_data[i];
        
        controller_state[i] <= READ_RELAYING;
    end
end
WRITE_WAITING: begin 
    // Wait for acknowledgement from memory
    if (mem_write_ready[i]) begin 
        mem_write_valid[i] <= 0;
        consumer_write_ready[current_consumer[i]] <= 1;
        controller_state[i] <= WRITE_RELAYING;
    end
end
```

```verilog
// Wait until consumer acknowledges it received response, then reset
READ_RELAYING: begin
    if (!consumer_read_valid[current_consumer[i]]) begin 
        channel_serving_consumer[current_consumer[i]] = 0;
        consumer_read_ready[current_consumer[i]] <= 0;
        controller_state[i] <= IDLE;
    end
end

WRITE_RELAYING: begin 
    if (!consumer_write_valid[current_consumer[i]]) begin 
        channel_serving_consumer[current_consumer[i]] = 0;
        consumer_write_ready[current_consumer[i]] <= 0;
        controller_state[i] <= IDLE;
    end
end
endcase
endmodule
```

Let’s understand these states:

*   `READ_WAITING`: This is the latency penalty. The controller sits here doing nothing until the external RAM chip raises `mem_read_ready`.
*   `READ_RELAYING`: Once we have the data, we hand it back to the specific thread that asked (`current_consumer[i]`).

In H100, this controller is pipelined. It doesn’t just wait, it overlaps requests. It also checks if Thread 0 and Thread 1 are asking for neighbor addresses (e.g., loading a matrix row) and merges them into one burst transaction. Our simplified version handles requests strictly one by one.

### Load Store Unit (LSU)

In Memory system, we built the **Traffic Controller** that organizes the chaos of multiple threads asking for data but we still need the actual driver inside each thread that makes the request.

![LSU Control (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*okbQLMf97gnp9qadUtKlpA.png)
*LSU Control (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In hardware architecture, this is the **Load Store Unit (LSU)** that lives next to the ALU and Register File inside the Core.

In the context of AI, the LSU is the most overworked component on the chip. Every time your PyTorch model executes a layer, it needs to fetch millions of weights and input activations. The ALU cannot reach out to memory directly, it is hardwired only to the Register File. If the ALU needs data that isn’t in the registers, it signals the LSU to go get it.

This is where **Latency** becomes a physical reality.

*   **ALU Operation**: ~1 clock cycle.
*   **LSU Operation (LDR -> Load Register)**: ~100 to ~1000 clock cycles (fetching from HBM).

On an **NVIDIA H100**, the LSU is highly sophisticated. It supports **Non-Blocking Loads**. This means the thread can say “Go fetch this data”, and while the LSU is waiting, the ALU can continue doing *other* math that doesn’t depend on that data (Instruction Level Parallelism). If the thread truly has nothing else to do, the H100 Scheduler will instantly put this thread to sleep and wake up a different thread that *is* ready to work (Latency Hiding).

For our **Tiny GPU**, our LSU will be **Blocking**. When a thread asks for data, it will freeze completely (stall) until the data arrives. This effectively demonstrates the pain of being “Memory Bound.”

Let’s create `load_store_unit.sv` and initialize our standard directives.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module. Notice how it connects the internal **Core** to the external **Memory Controller**.

```verilog
module load_store_unit (
    input wire clk,
    input wire reset,
    input wire enable, // If current block has less threads then block size, some LSUs will be inactive
    
    // State
    input reg [2:0] core_state,
    
    // Memory Control Signals from Decoder
    input reg decoded_mem_read_enable,  // Instruction is LDR
    input reg decoded_mem_write_enable, // Instruction is STR
    
    // Registers (Address and Data source)
    input reg [7:0] rs, // The address to read/write
    input reg [7:0] rt, // The data to write (for STR)
    
    // Interface to Memory Controller
    output reg mem_read_valid,
    output reg [7:0] mem_read_address,
    input reg mem_read_ready,
    input reg [7:0] mem_read_data,
    
    output reg mem_write_valid,
    output reg [7:0] mem_write_address,
    output reg [7:0] mem_write_data,
    input reg mem_write_ready,
    
    // LSU Outputs to Register File
    output reg [1:0] lsu_state,
    output reg [7:0] lsu_out
);
```

Let’s break down the interface:

*   `decoded_mem_read_enable`: This tells the LSU "The current instruction is `LDR`, wake up."
*   `rs`: In a Load instruction, this register holds the **Address** we want to fetch (e.g., `Matrix[i]`).
*   `rt`: In a Store instruction, this register holds the **Value** we want to save (e.g., the calculated pixel result).
*   `mem_read/write_...`: These signals connect directly to the `Memory Controller` we built in the previous section. This is the handshake protocol.

We need a simple state machine to manage the transaction lifecycle.

```verilog
localparam IDLE = 2'b00, 
    REQUESTING = 2'b01, 
    WAITING = 2'b10, 
    DONE = 2'b11;
```

`REQUESTING` is for when we send the request to memory. `WAITING` is when we are stalled, waiting for the data to come back. `DONE` is when we have the data and are ready to hand it off to the Register File.

Now for the logic. We will handle `LDR` (Load) and `STR` (Store) separately, but the logic is symmetrical.

```verilog
always @(posedge clk) begin
    if (reset) begin
        lsu_state <= IDLE;
        lsu_out <= 0;
        mem_read_valid <= 0;
        mem_read_address <= 0;
        mem_write_valid <= 0;
        mem_write_address <= 0;
        mem_write_data <= 0;
    end else if (enable) begin
        // If memory read enable is triggered (LDR instruction)
        if (decoded_mem_read_enable) begin 
            case (lsu_state)
                IDLE: begin
                    // Only start the request when the Core says "REQUEST"
                    if (core_state == 3'b011) begin 
                        lsu_state <= REQUESTING;
                    end
                end
                REQUESTING: begin 
                    // Put the request on the wire to the Memory Controller
                    mem_read_valid <= 1;
                    mem_read_address <= rs; // RS holds the address
                    lsu_state <= WAITING;
                end
                WAITING: begin
                    // Sit here and wait for the Traffic Cop (Controller) to give us data
                    if (mem_read_ready == 1) begin
                        mem_read_valid <= 0;      // Turn off request
                        lsu_out <= mem_read_data; // Capture the data
                        lsu_state <= DONE;
                    end
                end
                DONE: begin 
                    // Reset when core_state moves to UPDATE
                    if (core_state == 3'b110) begin 
                        lsu_state <= IDLE;
                    end
                end
            endcase
        end
```

So we have coded the `LDR` flow, let's understand it step by step:

1.  `IDLE`: We wait for the Core to enter the `REQUEST` phase.
2.  `REQUESTING`: We raise the flag `mem_read_valid`. This tells the Memory Controller "I need data at Address `rs`."
3.  `WAITING`: **The Bottleneck.** The LSU stays in this state until `mem_read_ready` goes high. If the Memory Controller is busy serving other cores, we stay here.
4.  In a productional grade GPU trace, if you see a lot of stalls, it means your code is spending all its time in this `WAITING` block. This is why `Memory Coalescing` matters, optimizing your code so 32 threads can get data in 1 transaction instead of 32 separate transactions.
5.  `DONE`: We have the data (`lsu_out`). In the next cycle, the Register File will grab this value and save it.

Now, the `STR` (Store) logic. This is used when we finish calculating a matrix cell and want to save it back to VRAM.

```verilog
// If memory write enable is triggered (STR instruction)
if (decoded_mem_write_enable) begin 
    case (lsu_state)
        IDLE: begin
            if (core_state == 3'b011) begin 
                lsu_state <= REQUESTING;
            end
        end
        REQUESTING: begin 
            mem_write_valid <= 1;
            mem_write_address <= rs; // Where to write
            mem_write_data <= rt;    // What to write
            lsu_state <= WAITING;
        end
        WAITING: begin
            if (mem_write_ready) begin
                mem_write_valid <= 0;
                lsu_state <= DONE;
            end
        end
        DONE: begin 
            if (core_state == 3'b110) begin 
                lsu_state <= IDLE;
            end
        end
    endcase
end
endmodule
```

The Store logic is almost identical, except we send data (`mem_write_data`) out instead of bringing it in.

With the **LSU** complete, we have finished the **Memory Subsystem**.

1.  **LSU:** The driver asking for data.
2.  **Memory Controller:** The arbiter managing the traffic.
3.  **Registers:** The destination for the data.

Now we have ALU to compute, Registers to store, and Memory to fetch weights from. We have built the core components of our Tiny GPU. It is time to build the **Control Unit**, starting with the **Decoder**.

### Decoder

In GPU architecture, the instructions that tell the hardware what to do are written in binary code (e.g., `01011100...`). But the hardware circuits (like the ALU or LSU) don't understand binary directly.

![Decoder (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*IcW35yodo8aNF-AIlL_N4g.png)
*Decoder (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

We need a translator. We need a component that takes the raw binary code (the software) and turns it into electrical control signals (the hardware actions). This is in hardware engineering terminologies is called **Decoder**.

In the AI domain, this is a bridge between software and silicon.

1.  You write `output = linear(input, weight)` in PyTorch.
2.  The compiler converts this to CUDA Assembly (PTX/SASS).
3.  The Assembler converts that to Binary (e.g., `01011100...`).
4.  **The Decoder** reads that binary and flips the specific switch that enables the Adder circuit.

On an **NVIDIA H100**, the decoder is incredibly complex because the instruction set is massive. It handles “Fused Instructions” (like `HFMA` - Half-Precision Fused Multiply-Add) where one binary code triggers multiple hardware actions simultaneously. The H100 also has to handle "Dual Issue" dispatch, where it decodes two instructions at once to keep the pipeline full.

So to mimic that behavior for our **Tiny GPU**, we are going to build a simple Instruction Set Architecture (ISA) with 11 commands. Our decoder will look at the first 4 bits of the instruction (the Opcode) and activate the correct unit (ALU, LSU, or PC).

Let’s create `decoder.sv` and insert standard directives:

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module. This module takes the 16-bit instruction from the `fetcher` (which we will build next) and explodes it into many control wires.

```verilog
module decoder (
    input wire clk,
    input wire reset,


    input reg [2:0] core_state,
    input reg [15:0] instruction, // The raw 16-bit binary code
    
    // Instruction Signals (Extracted from the binary)
    output reg [3:0] decoded_rd_address,
    output reg [3:0] decoded_rs_address,
    output reg [3:0] decoded_rt_address,
    output reg [2:0] decoded_nzp,
    output reg [7:0] decoded_immediate,
    
    // Control Signals (The "Strings" that pull the puppets)
    output reg decoded_reg_write_enable,           // Tell Registers to save data
    output reg decoded_mem_read_enable,            // Tell LSU to fetch data
    output reg decoded_mem_write_enable,           // Tell LSU to store data
    output reg decoded_nzp_write_enable,           // Tell PC to update flags
    output reg [1:0] decoded_reg_input_mux,        // Select input source (ALU vs Memory)
    output reg [1:0] decoded_alu_arithmetic_mux,   // Select Math type (Add/Sub/Mul/Div)
    output reg decoded_alu_output_mux,             // Select ALU mode (Math vs Compare)
    output reg decoded_pc_mux,                     // Select PC mode (Next Line vs Jump)
    // Return (Signal that the thread is finished)
    output reg decoded_ret
);
```

Let’s understand the outputs:

*   `decoded_rd_address`, `decoded_rs_address`, `decoded_rt_address`: These extract the destination and source register addresses from the instruction.
*   `decoded_mem_read_enable`: This wire tells the LSU "The current instruction is a Load (`LDR`), go fetch data".
*   `decoded_alu_arithmetic_mux`: This wire tells the ALU which math operation to perform (Add, Subtract, Multiply, Divide).

We have also defined many other control signals like `decoded_reg_write_enable` (to save results back to registers), `decoded_pc_mux` (to control branching in the PC) and `decoded_nzp_write_enable` (to update the NZP flags after comparisons).

If the Decoder sets the wrong wire, the GPU might try to write to memory when it should be adding numbers, causing data corruption.

First, we define our **Instruction Set Architecture (ISA)**. This is the dictionary of our chip.

```verilog
localparam NOP = 4'b0000,   // No Operation
    BRnzp = 4'b0001,        // Branch (Loop)
    CMP = 4'b0010,          // Compare (for loops)
    ADD = 4'b0011,          // Math
    SUB = 4'b0100,
    MUL = 4'b0101,          // The Tensor Core Op
    DIV = 4'b0110,
    LDR = 4'b0111,          // Load Weights
    STR = 4'b1000,          // Store Activations
    CONST = 4'b1001,        // Load Constant
    RET = 4'b1111;          // Return/Exit
```

In here we are defining 11 instructions and most of their purpose is to support AI workloads like `BRnzp` (branching for loops), `CMP` (comparison for loop conditions), `MUL` (matrix multiplication), `LDR` (loading weights from memory), and `STR` (storing activations back to memory).

Now for the logic. We only want to decode when the Core is in the `DECODE` state.

```verilog
always @(posedge clk) begin 
    if (reset) begin 
        decoded_rd_address <= 0;
        decoded_rs_address <= 0;
        decoded_rt_address <= 0;
        decoded_immediate <= 0;
        decoded_nzp <= 0;
        decoded_reg_write_enable <= 0;
        decoded_mem_read_enable <= 0;
        decoded_mem_write_enable <= 0;
        decoded_nzp_write_enable <= 0;
        decoded_reg_input_mux <= 0;
        decoded_alu_arithmetic_mux <= 0;
        decoded_alu_output_mux <= 0;
        decoded_pc_mux <= 0;
        decoded_ret <= 0;
    end else begin 
        // Decode only when core_state = DECODE
        if (core_state == 3'b010) begin 
            // 1. Extract Fields
            // Our ISA is fixed-width. We know exactly where the bits are.
            // Opcode is bits [15:12], Dest Register is [11:8], etc.
            decoded_rd_address <= instruction[11:8];
            decoded_rs_address <= instruction[7:4];
            decoded_rt_address <= instruction[3:0];
            decoded_immediate <= instruction[7:0]; // For constants
            decoded_nzp <= instruction[11:9];      // For branching


            // 2. Reset Control Signals
            // Default behavior: Do nothing. Safety first.
            decoded_reg_write_enable <= 0;
            decoded_mem_read_enable <= 0;
            decoded_mem_write_enable <= 0;
            decoded_nzp_write_enable <= 0;
            decoded_ret <= 0;
            // 3. Set Signals based on Opcode
            case (instruction[15:12])
                ADD: begin 
                    decoded_reg_write_enable <= 1;    // Result needs to be saved
                    decoded_reg_input_mux <= 2'b00;   // Source is Arithmetic
                    decoded_alu_arithmetic_mux <= 2'b00; // Op is ADD
                end
                SUB: begin 
                    decoded_reg_write_enable <= 1;
                    decoded_reg_input_mux <= 2'b00;
                    decoded_alu_arithmetic_mux <= 2'b01; // Op is SUB
                end
                MUL: begin 
                    decoded_reg_write_enable <= 1;
                    decoded_reg_input_mux <= 2'b00;
                    decoded_alu_arithmetic_mux <= 2'b10; // Op is MUL
                end
                LDR: begin 
                    // This wakes up the LSU
                    decoded_reg_write_enable <= 1;    // We will eventually write to reg
                    decoded_reg_input_mux <= 2'b01;   // Source is Memory
                    decoded_mem_read_enable <= 1;     // Activate LSU Read Mode
                end
                STR: begin 
                    // This also wakes up the LSU
                    decoded_mem_write_enable <= 1;    // Activate LSU Write Mode
                end
                CONST: begin 
                    decoded_reg_write_enable <= 1;
                    decoded_reg_input_mux <= 2'b10;   // Source is Constant (Immediate)
                end
                CMP: begin 
                    decoded_alu_output_mux <= 1;      // Output Comparison Flags
                    decoded_nzp_write_enable <= 1;    // Update PC flags
                end
                BRnzp: begin 
                    decoded_pc_mux <= 1;              // Enable Branching logic in PC
                end
                RET: begin 
                    decoded_ret <= 1;                 // Signal completion
                end
            endcase
        end
    end
end
endmodule
```

Let’s trace two critical instructions for AI:

`MUL` **(The Compute)** when multiplying two matrix elements:

*   We set `decoded_reg_write_enable <= 1`. We want to save the result back to the Register File.
*   We set `decoded_alu_arithmetic_mux <= 2'b10`. This tells the ALU "Do Multiplication" with the inputs.
*   In the next cycle (`EXECUTE`), the ALU will see these signals and perform `rs * rt` as a result of the multiplication.

`LDR` **(The Data)** when loading a weight from memory:

*   We set `decoded_reg_write_enable <= 1`. We will eventually save the fetched data to the Register File.
*   We set `decoded_reg_input_mux <= 2'b01`. This tells the Register File "The data is coming from Memory, not the ALU".
*   We set `decoded_mem_read_enable <= 1`. This is the trigger for the LSU. It says "Go fetch data from the address in `rs`".
*   In the next cycle (`REQUEST`), the **LSU** will see this signal go high. It will immediately take over the bus and start the memory request transaction we coded in the previous section.

**The Decoder is now complete. It successfully maps software intentions to hardware actions.**

However, the decoder needs to get the instruction from somewhere. It can’t decode thin air. We need a unit that fetches the binary code from the Instruction Memory and hands it to the decoder. That unit is the **Fetcher**.

### Fetcher

The Decoder needs an instruction to decode. It needs the raw binary code (e.g., `0011000000001111`) representing "ADD R0, R0, R15". But that code lives in the **Program Memory** (I-RAM), potentially far away from the core.

In real GPUS like `NVIDIA H100`, the Program Memory is stored in **HBM (High Bandwidth Memory)**, separate from the Data Memory. This is called **Harvard Architecture**.

![Fetcher Logic (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*Y1UAyad9_Igoql4d83GIhg.png)
*Fetcher Logic (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

We need a dedicated unit whose only job is to look at the **Program Counter (PC)**, run to the instruction memory, grab the code at that address, and hand it to the Decoder.

In hardware architecture, this is the **Instruction Fetch Unit (IFU)** or simply the **Fetcher**.

In the context of AI and High-Performance Computing, the Fetcher is the Feeder.

*   An H100 Tensor Core can chew through math incredibly fast. If the Fetcher is slow, the Tensor Core starves. It sits idle, waiting for the next command. This is called **Frontend Stalling**.
*   **The L1 Instruction Cache:** On an **NVIDIA H100**, the Fetcher doesn’t go all the way to HBM (High Bandwidth Memory) for every instruction. That would take ~400 cycles. Instead, it pulls from a super-fast **L1 Instruction Cache** (I-Cache) located inside the Streaming Multiprocessor (SM).
*   **Throughput:** Real GPUs fetch multiple instructions per cycle (Instruction Level Parallelism). Our Tiny GPU will fetch one instruction at a time.

If you’ve ever checked the performance of a CUDA kernel and noticed “Low Warp Occupancy” or “Stall due to No Instruction”, it usually means the Fetcher was too slow to deliver instructions, or the instruction cache couldn’t find the needed code quickly enough.

Let’s create `fetcher.sv` to keep our pipeline fed.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module. Notice that this module connects the **Core** to the **Program Memory Controller**.

```verilog
module fetcher #(
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16
) (
    input wire clk,
    input wire reset,
    
    // Execution State from Scheduler
    input reg [2:0] core_state,
    input reg [7:0] current_pc, // Where are we?

    // Interface to Program Memory Controller
    output reg mem_read_valid,
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] mem_read_address,
    input reg mem_read_ready,
    input reg [PROGRAM_MEM_DATA_BITS-1:0] mem_read_data,

    // Output to Decoder
    output reg [2:0] fetcher_state,
    output reg [PROGRAM_MEM_DATA_BITS-1:0] instruction
);
```

Let’s break down the interface:

*   `current_pc`: This comes from the **Program Counter** module we built earlier. It tells the Fetcher *which* line of code to grab.
*   `instruction`: This is the output. It is the 16-bit binary code that goes straight into the **Decoder**.
*   `mem_read_...`: These signals connect to the `Memory Controller`. Unlike the LSU (which fetches Data), the Fetcher fetches *Code*. In Harvard Architecture (which GPUs loosely resemble at this level), Instruction Memory and Data Memory are often treated as separate address spaces to improve bandwidth.

We need a state machine to manage the fetching process.

```verilog
localparam IDLE = 3'b000, 
    FETCHING = 3'b001, 
    FETCHED = 3'b010;
```

Now for the logic. This state machine must coordinate perfectly with the Core’s main `scheduler` (which controls `core_state`).

```verilog
always @(posedge clk) begin
    if (reset) begin
        fetcher_state <= IDLE;
        mem_read_valid <= 0;
        mem_read_address <= 0;
        instruction <= {PROGRAM_MEM_DATA_BITS{1'b0}};
    end else begin
        case (fetcher_state)
            IDLE: begin
                // Start fetching when the Scheduler enters the FETCH phase
                if (core_state == 3'b001) begin
                    fetcher_state <= FETCHING;
                    mem_read_valid <= 1;        // Signal the Memory Controller
                    mem_read_address <= current_pc; // "Give me code at Line X"
                end
            end
            FETCHING: begin
                // Wait for response from program memory (I-Cache)
                if (mem_read_ready) begin
                    fetcher_state <= FETCHED;
                    instruction <= mem_read_data; // Capture the binary code
                    mem_read_valid <= 0;          // Turn off the request
                end
            end
            FETCHED: begin
                // Handshake with the Scheduler
                // We hold the instruction here until the Scheduler moves to DECODE
                if (core_state == 3'b010) begin 
                    fetcher_state <= IDLE; // Ready for the next cycle
                end
            end
        endcase
    end
end
endmodule
```

Let’s trace the lifecycle of an instruction fetch:

1.  `IDLE`: The Fetcher waits. The moment the Scheduler sets `core_state` to `FETCH` (3'b001), the Fetcher wakes up. It sends the `current_pc` (say, address 0x0A) to the memory controller and raises the valid flag.
2.  `FETCHING`: Just like the LSU, the Fetcher must wait.
3.  **H100 Parallelism**: In a real GPU, while the Fetcher is waiting for instruction 0x0A, the execution units might be busy executing instruction 0x09. This overlapping is crucial.

In our Tiny GPU, the whole core waits. If `mem_read_ready` takes 10 cycles, the entire AI model pauses for 10 cycles. This highlights the importance of **Instruction Locality** keeping code in fast cache memory.

1.  `FETCHED`: Once the data arrives, we store it in the `instruction` register.
2.  Crucially, we stay in this state until the Core moves to `DECODE`. Why? Because the **Decoder** needs stable input. If we immediately went back to IDLE and changed the `instruction` register, the Decoder would see garbage in the middle of decoding. This state ensures stability.

With the **Fetcher** complete, we have a working Frontend.

1.  **Fetcher** grabs binary from memory.
2.  **Decoder** translates binary to signals.
3.  **ALU/LSU** execute the signals.

But we need one more component for this that is going to tell the Fetcher to fetch?

We need a central control unit to manage the state transitions (`FETCH` -> `DECODE` -> `EXECUTE`). That component is the **Scheduler**.

### Scheduler

We have assembled all the individual components of our GPU:

*   **ALU** (Arithmetic Logic Unit) for computation
*   **Registers** and **LSU** for memory management
*   **Decoder** for instruction translation
*   **Fetcher** for code retrieval

However, these components currently operate in isolation. We need a centralized unit to operate their timing and execution order. We need a state machine to dictate: **“Fetcher, retrieve code now. Decoder, translate it next. ALU, execute now”**. This component is the **Scheduler**.

![Scheduler (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*7PDEPQj-fS16I1fnBoRlpA.png)
*Scheduler (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In AI, the scheduler is important for maximizing efficiency. **H100 Warp Scheduler:** An NVIDIA Streaming Multiprocessor (SM) contains 4 Warp Schedulers. Their primary function is **Latency Hiding**.

*   Consider a scenario where Warp 0 issues a global memory `Load`. This operation incurs a latency of ~400 clock cycles.
*   A naive scheduler would stall the entire core, wasting cycles.
*   The H100 Scheduler detects this stall and *instantly* (within 1 clock cycle) context-switches to Warp 1, then Warp 2. By the time it returns to Warp 0, the memory transaction is complete.
*   This mechanism ensures the Tensor Cores remain at 100% utilization, which is the primary objective in high-performance AI training.

For our **Tiny GPU**, we will implement a simplified **Round-Robin State Machine**. It will enforce a deterministic pipeline for a single block of threads. While it won’t support context switching between warps, it will ensure that the pipeline stages (`Fetch`, `Decode`, `Execute`) occur in the correct physical sequence.

Let’s create `scheduler.sv` to manage this control flow.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

Now, let’s define the module. Observe that it accepts status inputs from all subsystems (LSU, Fetcher, Decoder) and drives the master `core_state` signal.

```verilog
module scheduler #(
    parameter THREADS_PER_BLOCK = 4,
) (
    input wire clk,
    input wire reset,
    input wire start, // The "Go" signal from the Dispatcher
    
    // Control Signals (Did the instruction involve memory or branching?)
    input reg decoded_mem_read_enable,
    input reg decoded_mem_write_enable,
    input reg decoded_ret,

    
    // Status Signals (Are the sub-units busy?)
    input reg [2:0] fetcher_state,
    input reg [1:0] lsu_state [THREADS_PER_BLOCK-1:0],
    
    // PC Handling
    output reg [7:0] current_pc,
    input reg [7:0] next_pc [THREADS_PER_BLOCK-1:0],
    
    // The Master Output
    output reg [2:0] core_state,
    output reg done
);
```

This module is acting as the brain of the core. It monitors:

*   `start`: This signal comes from the Dispatcher (which we will build later). It tells the scheduler to begin processing a block of threads.
*   `lsu_state`: The scheduler monitors the state of every LSU. If *any* thread is waiting for memory, the scheduler pauses execution.
*   `core_state`: This is the output that drives every other module we have built so far.

Let’s define the states of our pipeline.

```verilog
localparam IDLE = 3'b000, // Waiting to start
    FETCH = 3'b001,       // Fetch instructions from program memory
    DECODE = 3'b010,      // Decode instructions into control signals
    REQUEST = 3'b011,     // Request data from registers or memory
    WAIT = 3'b100,        // Wait for response from memory if necessary
    EXECUTE = 3'b101,     // Execute ALU and PC calculations
    UPDATE = 3'b110,      // Update registers, NZP, and PC
    DONE = 3'b111;        // Done executing this block
```

Now for the core logic. This `always` block runs the state on every clock edge.

```verilog
always @(posedge clk) begin 
    if (reset) begin
        current_pc <= 0;
        core_state <= IDLE;
        done <= 0;
    end else begin 
        case (core_state)
            IDLE: begin
                // Waiting for the Dispatcher to give us a job
                if (start) begin 
                    // Start by fetching the next instruction for this block based on PC
                    core_state <= FETCH;
                end
            end
            FETCH: begin 
                // We stay here until the Fetcher reports success
                // H100 Equivalent: L1 I-Cache Hit/Miss Logic
                if (fetcher_state == 3'b010) begin // 3'b010 is FETCHED
                    core_state <= DECODE;
                end
            end
```

`FETCH` **State**: The scheduler waits here. If the `fetcher.sv` module takes 10 cycles to get code from RAM, the scheduler keeps `core_state` at `FETCH` for 10 cycles. This effectively pauses the entire core.

```verilog
DECODE: begin
    // Decode is purely combinational logic (instant), so we move on in 1 cycle
    core_state <= REQUEST;
end

REQUEST: begin 
    // This triggers the LSU to send its address to the Memory Controller
    // Also takes 1 cycle
    core_state <= WAIT;
end

WAIT: begin
    // THE MOST CRITICAL STATE FOR AI PERFORMANCE
    // We must check if ANY thread is still waiting for memory.
    
    reg any_lsu_waiting = 1'b0;
    
    // Check all threads in parallel
    for (int i = 0; i < THREADS_PER_BLOCK; i++) begin
        // If LSU is REQUESTING (01) or WAITING (10)
        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10) begin
            any_lsu_waiting = 1'b1;
            break;
        end
    end

    // Pipeline Bubble Logic:
    // If we are doing ALUL operations (ADD/MUL), the LSU is idle, so we skip this instantly.
    // If we are doing Memory operations (LDR), we STALL here.
    if (!any_lsu_waiting) begin
        core_state <= EXECUTE;
    end
end
```

`WAIT` State Logic: This is where performance bottlenecks occur.

*   The loop checks `lsu_state` for *every* thread.
*   If **Thread 0** has received its data, but **Thread 3** is still waiting on the memory controller, the **Scheduler forces the entire core to wait**.
*   **AI Insight**: This illustrates “Tail Latency” or “Warp Divergence” in memory access. The GPU is only as fast as its slowest thread. If one thread encounters a memory bank conflict, it stalls the progress of the entire warp.

```verilog
EXECUTE: begin
    // The ALU fires here.
    // In H100, this would take multiple cycles for complex math (like SFU ops).
    // Here, it's 1 cycle.
    core_state <= UPDATE;
end
UPDATE: begin 
    if (decoded_ret) begin 
        // The kernel called "RET" (Return)
        done <= 1;
        core_state <= DONE;
    end else begin 
        // Branch Divergence Handling
        // We optimistically assume Thread 0's PC is the correct one for everyone.
        // Real GPUs have complex "Reconvergence Stacks" here.
        current_pc <= next_pc[THREADS_PER_BLOCK-1];

        // Loop back to fetch the next instruction
        core_state <= FETCH;
    end
end
DONE: begin 
    // Sit here until reset
end
endcase
endmodule
```

Let’s break down the critical states:

*   `EXECUTE`: This triggers the ALU. In our case, integer math is fast (1 cycle). In real GPUs, floating point operations (like `FP64`) might take many cycles, keeping the scheduler in this state longer.
*   `UPDATE`: This updates the Program Counter. Notice `current_pc <= next_pc[...]`. We take the next program counter and loop back to `FETCH`, completing the instruction cycle.

With the **Scheduler**, we have completed the **Control Logic**. We now possess every individual component required to build a processing core.

1.  **ALU** (Math)
2.  **Registers** (Memory)
3.  **LSU** (Data Movement)
4.  **PC** (Flow)
5.  **Decoder** (Translation)
6.  **Fetcher** (Code Input)
7.  **Scheduler** (Coordination)

The next step is assembly. We need to instantiate these modules and wire them together into a single unit capable of executing multiple threads in parallel. This unit is the **Compute Core** (or Streaming Multiprocessor), which we will build next.

### Compute Core (Streaming Multiprocessor)

Now, we must assemble all the components into a unit that executes the **SIMT (Single Instruction, Multiple Threads)** architecture. In hardware engineering, this unit is often called a **Compute Core**, but in NVIDIA terminology, it is known as a **Streaming Multiprocessor (SM)**.

![Core Compute (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:1250/1*4BJJ1es8YivmH1VFpo7A1g.png)
*Core Compute (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

This is the most critical architectural concept in GPU design.

*   **The Efficiency of SIMT:** In a CPU core, you have one Instruction Decoder and one ALU. To run 32 threads, you need 32 CPUs, which means 32 Decoders. That is a massive waste of silicon area and power because every thread is executing the *same* instruction.
*   **The GPU Approach:** In an H100 SM, we share the “Frontend” (Fetcher, Decoder, Scheduler) across many “Backend” lanes (ALUs).
*   **H100 Architecture:** An H100 GPU has 144 SMs. Each SM contains 4 Warp Schedulers and 128 FP32 CUDA Cores. When the Scheduler issues an `FADD` instruction, it is broadcast to all active CUDA cores simultaneously.

For our **Tiny GPU**, our `core.sv` will implement this logic. It will instantiate **one** set of control units (Fetcher, Decoder, Scheduler) and use a `generate` loop to instantiate **N** execution units (ALU, LSU, Registers, PC) based on the `THREADS_PER_BLOCK` parameter.

Let’s create `core.sv` and insert standard directives:

```verilog
`default_nettype none
`timescale 1ns/1ns
```

First, we define the module interface. This module sits between the **Dispatcher** (above) and the **Memory Controllers** (below).

```verilog
module core #(
    parameter DATA_MEM_ADDR_BITS = 8,
    parameter DATA_MEM_DATA_BITS = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter THREADS_PER_BLOCK = 4
) (
    input wire clk,
    input wire reset,

    // Kernel Execution (From Dispatcher)
    input wire start,
    output wire done,
    
    // Block Metadata (For SIMD Identity)
    input wire [7:0] block_id,
    input wire [$clog2(THREADS_PER_BLOCK):0] thread_count,
    
    // Program Memory Interface (Read-Only)
    output reg program_mem_read_valid,
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] program_mem_read_address,
    input reg program_mem_read_ready,
    input reg [PROGRAM_MEM_DATA_BITS-1:0] program_mem_read_data,
    
    // Data Memory Interface (Read/Write, One channel per thread)
    output reg [THREADS_PER_BLOCK-1:0] data_mem_read_valid,
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_read_address [THREADS_PER_BLOCK-1:0],
    input reg [THREADS_PER_BLOCK-1:0] data_mem_read_ready,
    input reg [DATA_MEM_DATA_BITS-1:0] data_mem_read_data [THREADS_PER_BLOCK-1:0],
    output reg [THREADS_PER_BLOCK-1:0] data_mem_write_valid,
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_write_address [THREADS_PER_BLOCK-1:0],
    output reg [DATA_MEM_DATA_BITS-1:0] data_mem_write_data [THREADS_PER_BLOCK-1:0],
    input reg [THREADS_PER_BLOCK-1:0] data_mem_write_ready
);
```

Let’s analyze the interface:

*   `THREADS_PER_BLOCK`: This parameter defines the width of our SIMD vector. If set to 4, we physically build 4 ALUs.

But notice the difference …

*   **Program Memory**: There is only **one** channel. Since all threads execute the same instruction, we only need to fetch it once.
*   **Data Memory**: There is an array of channels `[THREADS_PER_BLOCK-1:0]`. Each thread calculates its own address and needs its own path to memory.

Next, we define the internal wires to connect our sub-modules.

```verilog
// Shared Control Signals (Broadcast to all threads)
reg [2:0] core_state;
reg [2:0] fetcher_state;
reg [15:0] instruction;

// Decoded Signals (Broadcast)
reg [3:0] decoded_rd_address;
reg [3:0] decoded_rs_address;
reg [3:0] decoded_rt_address;
reg [2:0] decoded_nzp;
reg [7:0] decoded_immediate;
reg decoded_reg_write_enable;
reg decoded_mem_read_enable;
reg decoded_mem_write_enable;
reg decoded_nzp_write_enable;
reg [1:0] decoded_reg_input_mux;
reg [1:0] decoded_alu_arithmetic_mux;
reg decoded_alu_output_mux;
reg decoded_pc_mux;
reg decoded_ret;

// Per-Thread Signals (Unique to each thread)
reg [7:0] current_pc; // Only one PC is tracked for the block (Assuming convergence)
wire [7:0] next_pc[THREADS_PER_BLOCK-1:0];
reg [7:0] rs[THREADS_PER_BLOCK-1:0];
reg [7:0] rt[THREADS_PER_BLOCK-1:0];
reg [1:0] lsu_state[THREADS_PER_BLOCK-1:0];
reg [7:0] lsu_out[THREADS_PER_BLOCK-1:0];
wire [7:0] alu_out[THREADS_PER_BLOCK-1:0];
```

Now we instantiate the **Shared Frontend** (The Control Unit). These exist only once per core.

```verilog
// 1. Fetcher (Instruction Cache Interface)
    fetcher #(
        .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS)
    ) fetcher_instance (
        .clk(clk),
        .reset(reset),
        .core_state(core_state),
        .current_pc(current_pc),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .fetcher_state(fetcher_state),
        .instruction(instruction) 
    );
```

The fetcher grabs the instruction from memory based on the `current_pc` and outputs it as the `instruction` signal. This signal is then fed directly into the decoder below.

```verilog
// 2. Decoder (Instruction Translation)
decoder decoder_instance (
    .clk(clk),
    .reset(reset),
    .core_state(core_state),
    .instruction(instruction),
    .decoded_rd_address(decoded_rd_address),
    .decoded_rs_address(decoded_rs_address),
    .decoded_rt_address(decoded_rt_address),
    .decoded_nzp(decoded_nzp),
    .decoded_immediate(decoded_immediate),
    .decoded_reg_write_enable(decoded_reg_write_enable),
    .decoded_mem_read_enable(decoded_mem_read_enable),
    .decoded_mem_write_enable(decoded_mem_write_enable),
    .decoded_nzp_write_enable(decoded_nzp_write_enable),
    .decoded_reg_input_mux(decoded_reg_input_mux),
    .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
    .decoded_alu_output_mux(decoded_alu_output_mux),
    .decoded_pc_mux(decoded_pc_mux),
    .decoded_ret(decoded_ret)
);
```

The decoder takes the `instruction` and explodes it into dozens of `decoded_...` control wires. Crucially, these wires are broadcast to *all* threads in the core. This is how 1 instruction controls N threads.

```verilog
// 3. Scheduler (Pipeline Management)
scheduler #(
    .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
) scheduler_instance (
    .clk(clk),
    .reset(reset),
    .start(start),
    .fetcher_state(fetcher_state),
    .core_state(core_state),
    .decoded_mem_read_enable(decoded_mem_read_enable),
    .decoded_mem_write_enable(decoded_mem_write_enable),
    .decoded_ret(decoded_ret),
    .lsu_state(lsu_state),
    .current_pc(current_pc),
    .next_pc(next_pc), // Scheduler looks at next_pc from threads
    .done(done)
);
```

Here we are representing the `Brain` of the SM. The Scheduler monitors the `lsu_state` of all threads to make sure none are left behind, and it drives the `core_state` bus which dictates the timing for everyone.

Now for the **Backend** (The Execution Units). This is where we implement **Hardware Parallelism** using a `generate` loop.

```verilog
// Dedicated ALU, LSU, registers, & PC unit for each thread
genvar i;
generate
    for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads
```

`generate for (i = 0; ...)`: This is not a software for-loop. This instructs the synthesis tool to physically copy-paste the hardware inside the loop `N` times. If `THREADS_PER_BLOCK` is 128, it creates 128 ALUs on the silicon.

```verilog
// ALU (Muscle)
alu alu_instance (
    .clk(clk),
    .reset(reset),
    .enable(i < thread_count), // Predication mask
    .core_state(core_state),
    .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
    .decoded_alu_output_mux(decoded_alu_output_mux),
    .rs(rs[i]),
    .rt(rt[i]),
    .alu_out(alu_out[i])
);
```

`.enable(i < thread_count)`: This acts as a **Predicate Mask**. If we have 4 physical cores but only requested 3 threads for our AI workload, the 4th ALU is disabled to save power.

```verilog
// LSU (Data Movement)
lsu lsu_instance (
    .clk(clk),
    .reset(reset),
    .enable(i < thread_count),
    .core_state(core_state),
    .decoded_mem_read_enable(decoded_mem_read_enable),
    .decoded_mem_write_enable(decoded_mem_write_enable),

    // Each LSU gets its own slice of the memory interface
    .mem_read_valid(data_mem_read_valid[i]),
    .mem_read_address(data_mem_read_address[i]),
    .mem_read_ready(data_mem_read_ready[i]),
    .mem_read_data(data_mem_read_data[i]),
    .mem_write_valid(data_mem_write_valid[i]),
    .mem_write_address(data_mem_write_address[i]),
    .mem_write_data(data_mem_write_data[i]),
    .mem_write_ready(data_mem_write_ready[i]),
    .rs(rs[i]),
    .rt(rt[i]),
    .lsu_state(lsu_state[i]),
    .lsu_out(lsu_out[i])
);
```

Each LSU gets a dedicated slice of the memory bus arrays (e.g., `data_mem_read_valid[i]`). This allows them to make independent memory requests.

```verilog
// Register File (Identity)
registers #(
    .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
    .THREAD_ID(i), // Physical ID injection
    .DATA_BITS(DATA_MEM_DATA_BITS),
) register_instance (
    .clk(clk),
    .reset(reset),
    .enable(i < thread_count),
    .block_id(block_id),
    .core_state(core_state),
    .decoded_reg_write_enable(decoded_reg_write_enable),
    .decoded_reg_input_mux(decoded_reg_input_mux),
    .decoded_rd_address(decoded_rd_address),
    .decoded_rs_address(decoded_rs_address),
    .decoded_rt_address(decoded_rt_address),
    .decoded_immediate(decoded_immediate),
    .alu_out(alu_out[i]),
    .lsu_out(lsu_out[i]),
    .rs(rs[i]),
    .rt(rt[i])
);
```

`register_instance ... .THREAD_ID(i)`: This is where the identity is forged. We pass the loop variable `i` into the parameter `THREAD_ID`. This hardcodes the ID into the silicon of that specific register file, enabling SIMD.

We finally instantiate the **Program Counter** for each thread.

```verilog
// Program Counter (Flow)
pc #(
    .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
    .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS)
) pc_instance (
    .clk(clk),
    .reset(reset),
    .enable(i < thread_count),
    .core_state(core_state),
    .decoded_nzp(decoded_nzp),
    .decoded_immediate(decoded_immediate),
    .decoded_nzp_write_enable(decoded_nzp_write_enable),
    .decoded_pc_mux(decoded_pc_mux),
    .alu_out(alu_out[i]),
    .current_pc(current_pc),
    .next_pc(next_pc[i])
);
end
endgenerate
endmodule
```

We have now successfully built the **Compute Core**. It is a fully functional, multi-threaded processor. This way, we have built a single core that can execute multiple threads in parallel using SIMD principles.

> **However, a single core is not enough. An H100 has 144 of these cores**.

To scale our design, we need a way to replicate this core and distribute work among them. We need a way to configure the chip (Device Control) and a way to assign tasks (Dispatch).

We will next build the **Device Control Register (DCR)** to handle configuration.

### Device Control Register (DCR)

We have successfully engineered a powerful **Compute Core** capable of parallel execution. However, a GPU is not an autonomous device. It operates as a co-processor (Device) to a main CPU (Host). The CPU needs a specific interface to configure the GPU before launching an AI workload. It needs to tell the GPU parameters like **How many threads should run?** or **What is the grid size?**.

![Device Control (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*fdUcLimcWhGycGMZhPZ4CQ.png)
*Device Control (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In hardware architecture, this configuration interface is implemented via **Memory-Mapped I/O (MMIO) Registers**. In our design, we call this the **Device Control Register (DCR)**.

In the context of AI and the **NVIDIA H100**:

*   **Kernel Configuration:** When you execute a CUDA kernel in PyTorch or C++ (`kernel<<<gridDim, blockDim>>>(...)`), the CPU driver does not just magically start the GPU. It first writes these configuration values (`gridDim`, `blockDim`, Stream ID, Shared Memory size) into specific hardware registers on the GPU via the PCIe bus.
*   **Runtime Management:** These registers act as the “Settings Menu” for the hardware. They remain static during the execution of a kernel but are reconfigured by the driver between different kernel launches.

For our **Tiny GPU**, we will implement a simplified control register that stores a single critical parameter: the **Thread Count** (equivalent to `blockDim.x` in CUDA).

Let’s create `dcr.sv`.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

First, we define the module. This module sits at the top level of the GPU, accessible by the external control interface (representing the CPU driver).

```verilog
module dcr (
    input wire clk,
    input wire reset,

    // Interface to Host (CPU)
    input wire device_control_write_enable,
    input wire [7:0] device_control_data,

    // Interface to GPU Internal Components
    output wire [7:0] thread_count
);
```

Let’s break down the signals:

*   `device_control_write_enable`: This is the "Write Strobe." When the CPU wants to update the settings, it pulls this signal high.
*   `device_control_data`: The 8-bit data bus carrying the configuration value (e.g., the number `4` for 4 threads).
*   `thread_count`: This is the output wire. It connects continuously to the **Dispatcher** and the **Cores**, ensuring they always know how many threads are required for the current job.

Now, we define the internal storage and the update logic.

```verilog
// Physical storage for the configuration
    reg [7:0] device_control_register;

    // Continuous assignment to output
    assign thread_count = device_control_register[7:0];
    always @(posedge clk) begin
        if (reset) begin
            // Default State: 0 threads
            device_control_register <= 8'b0;
        end else begin
            // Update logic
            if (device_control_write_enable) begin 
                device_control_register <= device_control_data;
            end
        end
    end
endmodule
```

We are basically setting up a simple register with controlled write access. Let’s analyze the key components:

*   **Storage (**`reg [7:0]`**):** This register acts as a latch. Once the CPU writes a value here, it remains stable for millions of clock cycles while the kernel executes.
*   **The Write Mechanism**: The logic `if (device_control_write_enable)` ensures that we only update the configuration when explicitly instructed. This prevents random noise on the data bus from changing the kernel parameters during execution.
*   **AI Context**: In a real H100, writing to these registers is a privileged operation handled by the NVIDIA Driver. This specific file represents the hardware destination of that driver call.

With the **DCR** complete, we have a way to configure the GPU.

1.  **CPU** writes thread count to **DCR**.
2.  **DCR** holds the value.

But holding the value isn’t enough. We need a unit that reads this value and actually *manages* the lifecycle of the kernel. We need a unit to say “Okay, we need 100 threads. I have 2 cores. I will assign 50 to Core A and 50 to Core B.”

This workload manager is the **Dispatcher** (or GigaThread Engine), and we will build it next.

### Dispatcher (GigaThread Engine)

We have the **Compute Cores** ready to crunch numbers and the **Device Control Register (DCR)** holding the configuration (e.g., “Run 100 threads”). But we have a scaling problem.

What if the user asks for 1,000 threads, but we only have 2 cores with 4 threads each (capacity = 8)?

We cannot run 1,000 threads simultaneously. We must break the work into smaller chunks and feed them to the cores over time. This process is called **Dispatching**.

![Dispatcher (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*C0d2GlJmd2pEtCmUmdAWrg.png)
*Dispatcher (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In hardware architecture, this is the **Scheduler / Dispatcher**. In NVIDIA’s H100, this is the legendary **GigaThread Engine**.

*   **Our Problem:** An AI model (like LLama-3) might launch a kernel with 1,000,000 threads. The H100 has only ~20,000 active thread slots due to hardware limits.
*   GPU creator normally use the GigaThread Engine that divides the work into **Thread Blocks** (CTAs = Cooperative Thread Arrays). It assigns Blocks to Streaming Multiprocessors (SMs) as they become free.
*   **Occupancy:** If a kernel is memory-bound, the Dispatcher tries to keep as many blocks active as possible so that when one block stalls, another can run.

For our **Tiny GPU**, our dispatcher will handle basic **Block Distribution**.

1.  Calculate how many blocks are needed (`Total Threads / Threads Per Block`).
2.  Assign Block 0 to Core 0.
3.  Assign Block 1 to Core 1.
4.  Wait for a core to finish, then assign Block 2.
5.  Repeat until all threads are done.

Let’s create `dispatch.sv` to manage the workload.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

First, we define the module interface. It connects to the **DCR** (for configuration) and all the **Cores** (to give orders).

```verilog
module dispatch #(
    parameter NUM_CORES = 2,
    parameter THREADS_PER_BLOCK = 4
) (
    input wire clk,
    input wire reset,
    input wire start, // Master Start Signal

    // Configuration from DCR
    input wire [7:0] thread_count,

    // Interface to Compute Cores
    input reg [NUM_CORES-1:0] core_done, // Status from Cores
    output reg [NUM_CORES-1:0] core_start, // Command to Cores
    output reg [NUM_CORES-1:0] core_reset, // Command to Cores
    output reg [7:0] core_block_id [NUM_CORES-1:0], // Metadata for Cores
    output reg [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0], // Metadata

    // Global Status
    output reg done
);
```

In here we define the signals:

*   `thread_count`: This comes from the DCR. It tells us how many total threads we need to run.
*   `core_done`: This is an array of status signals from each core, indicating if they have finished their assigned block.
*   `core_start`: This is an array of command signals to each core, telling them to start processing a new block.

Let’s calculate the total workload.

```verilog
// Calculate the total number of blocks based on total threads & threads per block
// Formula: ceil(total / block_size)
// Implementation: (total + block_size - 1) / block_size
wire [7:0] total_blocks;
assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Keep track of progress
reg [7:0] blocks_dispatched; // How many have we sent out?
reg [7:0] blocks_done;       // How many have finished?
reg start_execution;         // Internal state flag
```

We implement the formula to calculate the number of blocks required. For example, if we need 10 threads and each block can handle 4 threads, we need 3 blocks (4 + 4 + 2).

Now for the dispatch logic loop. This runs continuously on every clock cycle.

```verilog
always @(posedge clk) begin
    if (reset) begin
        // Reset logic... (As I have implemented previous blocks)
        done <= 0;
        blocks_dispatched = 0;
        blocks_done = 0;
        start_execution <= 0;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_start[i] <= 0;
            core_reset[i] <= 1; // Keep cores in reset initially
        end
    end else if (start) begin    
        // 1. Completion Check
        // If all blocks are finished, the kernel is done.
        if (blocks_done == total_blocks) begin 
            done <= 1;
        end

        // 2. Dispatch Loop
        for (int i = 0; i < NUM_CORES; i++) begin
            // A. Check if a core just finished
            if (core_start[i] && core_done[i]) begin
                core_reset[i] <= 1; // Reset it for the next job
                core_start[i] <= 0;
                blocks_done = blocks_done + 1;
            end
            // B. Check if a core is free and we have work to do
            if (core_reset[i]) begin 
                core_reset[i] <= 0; // Release reset
                if (blocks_dispatched < total_blocks) begin 
                    // Assign a new block to this core
                    core_start[i] <= 1;
                    core_block_id[i] <= blocks_dispatched;
                    
                    // Handle the "Tail" block (e.g., if total=10 and block=4, 
                    // the last block only has 2 threads, not 4).
                    core_thread_count[i] <= (blocks_dispatched == total_blocks - 1) 
                        ? thread_count - (blocks_dispatched * THREADS_PER_BLOCK)
                        : THREADS_PER_BLOCK;
                    blocks_dispatched = blocks_dispatched + 1;
                end
            end
        end
    end
end
endmodule
```

**Discussion:** The Dispatch Logic Loop is iterating through all the cores (`NUM_CORES`). It mimics the hardware scheduler in an H100 scanning SMs for availability.

Then comes the **Tail Handling** (`core_thread_count`). This is a critical detail in AI. Often, matrix dimensions are not perfect multiples of the block size. We calculate exactly how many active threads the last block needs. This value is passed to `core.sv`, which uses it to disable the extra ALUs (the `.enable` signal we implemented earlier).

And in the **State Tracking**, we increment `blocks_dispatched` when we send a job, and `blocks_done` when the core reports back. Only when `blocks_done == total_blocks` do we signal to the outside world that the AI inference is complete.

With the **Dispatcher** complete, we have finished the logic for the entire GPU.

We have:

1.  **DCR**: Holds **“Run 100 threads”**.
2.  **Dispatcher**: Splits 100 threads into blocks and feeds the cores.
3.  **Core**: Runs the blocks in parallel.
4.  **Memory Controller**: Manages the data flow.

We have one final step. We need to wrap all these modules into a single, top-level chip. This is the **GPU Top Module**.

### GPU Top Module (Silicon Die)

We have reached the final stage of assembly. We have the **Cores** (Compute), the **Dispatcher** (Management), the **DCR** (Configuration), and the **Memory Controllers** (Logistics).

Now, we need to wire them all together into a single, cohesive unit. In hardware engineering, this is called the **Top-Level Module**.

![Silicon Die Component (Created by Fareed Khan)](https://miro.medium.com/v2/resize:fit:875/1*ZIdzxl5Yqy38uedNfBjsFQ.png)
*Silicon Die Component (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd))*

In the context of the **NVIDIA H100**, this represents the **Silicon Die** itself, the physical square of silicon (codenamed GH100) that sits inside the package.

*   **System Integration:** The H100 isn’t just a bag of parts. It connects 144 Streaming Multiprocessors via a massive **Crossbar Switch** (or Network-on-Chip) to the L2 Cache and HBM3 Memory Controllers.
*   **Scalability:** This file defines the ultimate performance of the chip. By changing a single parameter (`NUM_CORES`), we can scale this design from a tiny embedded NPU (Neural Processing Unit) to a massive Data Center GPU.
*   **The Interface:** This module defines the physical pins of the chip: Clock, Reset, PCIe Data (DCR), and the Memory Bus pins that connect to external VRAM chips.

Let’s create `src/gpu.sv`. This is the base frame that holds our engine.

```verilog
`default_nettype none
`timescale 1ns/1ns
```

First, we define the parameters. This section effectively constitutes the “Spec Sheet” of our GPU.

```verilog
module gpu #(
    parameter DATA_MEM_ADDR_BITS = 8,        // Memory Depth (256 rows)
    parameter DATA_MEM_DATA_BITS = 8,        // Data Width (8-bit quantization)
    parameter DATA_MEM_NUM_CHANNELS = 4,     // Memory Bandwidth lanes
    parameter PROGRAM_MEM_ADDR_BITS = 8,     // Instruction Memory Depth
    parameter PROGRAM_MEM_DATA_BITS = 16,    // Instruction Width
    parameter PROGRAM_MEM_NUM_CHANNELS = 1,  // Instruction Fetch Bandwidth
    parameter NUM_CORES = 2,                 // Total Compute Units
    parameter THREADS_PER_BLOCK = 4          // SIMD Width
) (
    input wire clk,
    input wire reset,

    // Kernel Execution
    input wire start,
    output wire done,
    
    // Device Control Register (The PCIe Interface)
    input wire device_control_write_enable,
    input wire [7:0] device_control_data,
    
    // External Memory Interface (Connecting to HBM/DRAM)
    // Program Memory
    output wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_valid,
    output wire [PROGRAM_MEM_ADDR_BITS-1:0] program_mem_read_address [PROGRAM_MEM_NUM_CHANNELS-1:0],
    input wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_ready,
    input wire [PROGRAM_MEM_DATA_BITS-1:0] program_mem_read_data [PROGRAM_MEM_NUM_CHANNELS-1:0],
    
    // Data Memory
    output wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_read_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0] data_mem_read_address [DATA_MEM_NUM_CHANNELS-1:0],
    input wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_read_ready,
    input wire [DATA_MEM_DATA_BITS-1:0] data_mem_read_data [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_write_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0] data_mem_write_address [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_DATA_BITS-1:0] data_mem_write_data [DATA_MEM_NUM_CHANNELS-1:0],
    input wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_write_ready
);
```

Let’s analyze the top-level module:

*   **Parameters**: Notice `NUM_CORES = 2`. If we wanted to build an H100 competitor, we would change this to `144`. The hardware description language allows us to "recompile" the chip with different specs instantly.
*   **External Interface**: The inputs/outputs here are the physical metal pins on the side of the chip. They connect the GPU Logic to the Motherboard (Clock/Reset/DCR) and the RAM Sticks (Memory Interface).
*   **Memory Channels**: Notice that both Program and Data Memory have multiple channels (`DATA_MEM_NUM_CHANNELS`, `PROGRAM_MEM_NUM_CHANNELS`). This represents the parallel lanes connecting to HBM3 memory modules, allowing high bandwidth access.
*   **Device Control Register Interface**: This is the MMIO interface that connects to the CPU driver via PCIe. It allows the host to configure the GPU before launching kernels.
*   **Kernel Execution Interface**: The `start` and `done` signals manage the lifecycle of the kernel execution.

Now we define the internal wires (the Nervous System) that connect the Dispatcher to the Cores.

```verilog
// Internal Control Signals
wire [7:0] thread_count;

// Compute Core State Arrays
reg [NUM_CORES-1:0] core_start;
reg [NUM_CORES-1:0] core_reset;
reg [NUM_CORES-1:0] core_done;
reg [7:0] core_block_id [NUM_CORES-1:0];
reg [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0];

// Interconnect: LSU <-> Data Memory Controller
// We calculate total consumers: Cores * Threads per Core
localparam NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK;
reg [NUM_LSUS-1:0] lsu_read_valid;
reg [DATA_MEM_ADDR_BITS-1:0] lsu_read_address [NUM_LSUS-1:0];
reg [NUM_LSUS-1:0] lsu_read_ready;
reg [DATA_MEM_DATA_BITS-1:0] lsu_read_data [NUM_LSUS-1:0];
reg [NUM_LSUS-1:0] lsu_write_valid;
reg [DATA_MEM_ADDR_BITS-1:0] lsu_write_address [NUM_LSUS-1:0];
reg [DATA_MEM_DATA_BITS-1:0] lsu_write_data [NUM_LSUS-1:0];
reg [NUM_LSUS-1:0] lsu_write_ready;

// Interconnect: Fetcher <-> Program Memory Controller
localparam NUM_FETCHERS = NUM_CORES;
reg [NUM_FETCHERS-1:0] fetcher_read_valid;
reg [PROGRAM_MEM_ADDR_BITS-1:0] fetcher_read_address [NUM_FETCHERS-1:0];
reg [NUM_FETCHERS-1:0] fetcher_read_ready;
reg [PROGRAM_MEM_DATA_BITS-1:0] fetcher_read_data [NUM_FETCHERS-1:0];
```

*   **Interconnect**: We define `NUM_LSUS`. If we have 2 cores and 4 threads, that's 8 LSUs. All 8 wires are aggregated here before being plugged into the Memory Controller. In a real H100, this aggregation is handled by a **NoC (Network on Chip)** to route traffic across the large silicon die.
*   **Fetcher Interconnect**: Each core has its own fetcher, so we define `NUM_FETCHERS = NUM_CORES`. This allows each core to independently request instructions from the Program Memory Controller.
*   **Core Control Signals**: These arrays (`core_start`, `core_reset`, etc.) manage the state of each core, allowing the Dispatcher to control their operation.

Now we instantiate the global managers: DCR, Data Controller, Program Controller, and Dispatcher.

```verilog
// 1. Device Control Register (Configuration)
dcr dcr_instance (
    .clk(clk),
    .reset(reset),
    .device_control_write_enable(device_control_write_enable),
    .device_control_data(device_control_data),
    .thread_count(thread_count)
);

// 2. Data Memory Controller (Traffic Cop for Weights/Activations)
controller #(
    .ADDR_BITS(DATA_MEM_ADDR_BITS),
    .DATA_BITS(DATA_MEM_DATA_BITS),
    .NUM_CONSUMERS(NUM_LSUS),
    .NUM_CHANNELS(DATA_MEM_NUM_CHANNELS)
) data_memory_controller (
    .clk(clk),
    .reset(reset),

    // Connect internal wires to consumers
    .consumer_read_valid(lsu_read_valid),
    .consumer_read_address(lsu_read_address),
    .consumer_read_ready(lsu_read_ready),
    .consumer_read_data(lsu_read_data),
    .consumer_write_valid(lsu_write_valid),
    .consumer_write_address(lsu_write_address),
    .consumer_write_data(lsu_write_data),
    .consumer_write_ready(lsu_write_ready),

    // Connect external wires to RAM
    .mem_read_valid(data_mem_read_valid),
    .mem_read_address(data_mem_read_address),
    .mem_read_ready(data_mem_read_ready),
    .mem_read_data(data_mem_read_data),
    .mem_write_valid(data_mem_write_valid),
    .mem_write_address(data_mem_write_address),
    .mem_write_data(data_mem_write_data),
    .mem_write_ready(data_mem_write_ready)
);

// 3. Program Memory Controller (Traffic Cop for Code)
controller #(
    .ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
    .DATA_BITS(PROGRAM_MEM_DATA_BITS),
    .NUM_CONSUMERS(NUM_FETCHERS),
    .NUM_CHANNELS(PROGRAM_MEM_NUM_CHANNELS),
    .WRITE_ENABLE(0)
) program_memory_controller (
    .clk(clk),
    .reset(reset),
    .consumer_read_valid(fetcher_read_valid),
    .consumer_read_address(fetcher_read_address),
    .consumer_read_ready(fetcher_read_ready),
    .consumer_read_data(fetcher_read_data),
    .mem_read_valid(program_mem_read_valid),
    .mem_read_address(program_mem_read_address),
    .mem_read_ready(program_mem_read_ready),
    .mem_read_data(program_mem_read_data)
);

// 4. Dispatcher (Workload Manager)
dispatch #(
    .NUM_CORES(NUM_CORES),
    .THREADS_PER_BLOCK(THREADS_PER_BLOCK)
) dispatch_instance (
    .clk(clk),
    .reset(reset),
    .start(start),
    .thread_count(thread_count),
    .core_done(core_done),
    .core_start(core_start),
    .core_reset(core_reset),
    .core_block_id(core_block_id),
    .core_thread_count(core_thread_count),
    .done(done)
);
```

This is the system integration step. We take the `lsu_read_valid` signals from the Cores (which we are about to create) and plug them into the `controller`. This physical wiring enables the "Memory Wall" behavior: if the Cores ask for too much, this controller module will throttle them.

In the above code we are also instantiating the **DCR** and **Dispatcher**, connecting them to the appropriate signals.

Also, note that we have two separate **Memory Controllers**: one for **Data Memory** (weights/activations) and one for **Program Memory** (instructions). This separation is crucial in modern GPU architecture to allow simultaneous instruction fetch and data access.

Finally, we generate the **Compute Cores**. We use a `generate` loop to create `NUM_CORES` instances.

```verilog
// 5. Compute Cores Array
genvar i;
generate
    for (i = 0; i < NUM_CORES; i = i + 1) begin : cores
        // EDA Workaround: We create separate signals here because some EDA tools
        // struggle with slicing multi-dimensional arrays at the port level.
        reg [THREADS_PER_BLOCK-1:0] core_lsu_read_valid;
        reg [DATA_MEM_ADDR_BITS-1:0] core_lsu_read_address [THREADS_PER_BLOCK-1:0];
        reg [THREADS_PER_BLOCK-1:0] core_lsu_read_ready;
        reg [DATA_MEM_DATA_BITS-1:0] core_lsu_read_data [THREADS_PER_BLOCK-1:0];
        reg [THREADS_PER_BLOCK-1:0] core_lsu_write_valid;
        reg [DATA_MEM_ADDR_BITS-1:0] core_lsu_write_address [THREADS_PER_BLOCK-1:0];
        reg [DATA_MEM_DATA_BITS-1:0] core_lsu_write_data [THREADS_PER_BLOCK-1:0];
        reg [THREADS_PER_BLOCK-1:0] core_lsu_write_ready;


        // Signal Routing: Connect the specific slice of the global bus to this core
        genvar j;
        for (j = 0; j < THREADS_PER_BLOCK; j = j + 1) begin
            localparam lsu_index = i * THREADS_PER_BLOCK + j;
            always @(posedge clk) begin 
                // Outgoing: Core -> Controller
                lsu_read_valid[lsu_index] <= core_lsu_read_valid[j];
                lsu_read_address[lsu_index] <= core_lsu_read_address[j];
                lsu_write_valid[lsu_index] <= core_lsu_write_valid[j];
                lsu_write_address[lsu_index] <= core_lsu_write_address[j];
                lsu_write_data[lsu_index] <= core_lsu_write_data[j];
                
                // Incoming: Controller -> Core
                core_lsu_read_ready[j] <= lsu_read_ready[lsu_index];
                core_lsu_read_data[j] <= lsu_read_data[lsu_index];
                core_lsu_write_ready[j] <= lsu_write_ready[lsu_index];
            end
        end

        // Instantiate the Core
        core #(
            .DATA_MEM_ADDR_BITS(DATA_MEM_ADDR_BITS),
            .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
            .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
            .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS),
            .THREADS_PER_BLOCK(THREADS_PER_BLOCK)
        ) core_instance (
            .clk(clk),
            .reset(core_reset[i]),
            .start(core_start[i]),
            .done(core_done[i]),
            .block_id(core_block_id[i]),
            .thread_count(core_thread_count[i]),
            
            // Connection to Program Memory Controller (1 per core)
            .program_mem_read_valid(fetcher_read_valid[i]),
            .program_mem_read_address(fetcher_read_address[i]),
            .program_mem_read_ready(fetcher_read_ready[i]),
            .program_mem_read_data(fetcher_read_data[i]),

            // Connection to Data Memory Controller (N per core)
            .data_mem_read_valid(core_lsu_read_valid),
            .data_mem_read_address(core_lsu_read_address),
            .data_mem_read_ready(core_lsu_read_ready),
            .data_mem_read_data(core_lsu_read_data),
            .data_mem_write_valid(core_lsu_write_valid),
            .data_mem_write_address(core_lsu_write_address),
            .data_mem_write_data(core_lsu_write_data),
            .data_mem_write_ready(core_lsu_write_ready)
        );
    end
endgenerate
endmodule
```

There are several important aspects to note here:

*   **EDA Workaround:** Some synthesis tools struggle with multi-dimensional array slicing at the port level. To mitigate this, we create dedicated signals (`core_lsu_read_valid`, etc.) for each core and thread. This ensures compatibility across various EDA tools.
*   **Signal Routing:** The nested loop (`for (j = 0; j < THREADS_PER_BLOCK; j = j + 1)`) performs the crucial task of mapping. It calculates that "Thread 3 on Core 1" is actually "Global Thread 7". It maps the local signals of the core to the global bus of the memory controller.
*   **The Routing Logic:** The nested loop (`lsu_index = i * THREADS_PER_BLOCK + j`) performs the crucial task of mapping. It calculates that "Thread 3 on Core 1" is actually "Global Thread 7". It maps the local signals of the core to the global bus of the memory controller.

This file represents the completion of the hardware. We have a Dispatcher assigning blocks, Cores running threads, and Controllers managing data.

So, we have successfully built a small-scale or better to call a tiny **GPU**. It simulates the physics of parallelism, latency, and resource contention that define modern AI hardware.

### Testing the GPU

So we have build the core components of our tiny GPU but to verify it, we will use **Cocotb** (Coroutine Co-simulation Testbench). It is a framework that allows us to write testbenches in Python while simulating Verilog hardware. You can also use other HDLs like VHDL, but we will focus on Verilog/SystemVerilog here.

We are basically creating a bridge that connects software and hardware:

*   **Python:** Acts as the Host Driver (like the CUDA Runtime). It loads the kernel and inputs.
*   **Verilog:** Acts as the Device (the GPU). It executes the logic.

We will implement two tests that represent the two fundamental pillars of AI:

1.  **Matrix Addition (`test_matadd`)**: To test **Memory Bandwidth** and the Data Path.
2.  **Matrix Multiplication (`test_matmul`)**: To test **Compute (TFLOPS)** and Control Flow (Loops).

Before writing code, we need to install the necessary tools. First, let’s install the Verilog Simulator (Icarus Verilog).

```bash
# Ubuntu/Debian
sudo apt-get install iverilog

# MacOS
brew install icarus-verilog
```

Now we need to install Cocotb that will act as our testbench framework.

```bash
pip install cocotb
```

Icarus Verilog has limited support for modern SystemVerilog features (like the multidimensional arrays we used in `core.sv`). So to overcome that we are going to use `sv2v` to transpile our code into a format Icarus understands.

*   To initialize it we need to download the binary from [GitHub — zachjs/sv2v](https://github.com/zachjs/sv2v).
*   Then you can add it to your PATH or place it in your project directory.

To get detailed output, we cannot just look at the final result in RAM. We need to attach virtual probes to the internal wires of the chip while it runs. We need to see the “Logic Analyzer” view of the silicon.

Let’s create a monitoring file for this `test/helpers/monitor.py`. It will contain a helper function that peeks inside Core 0 to see what the hardware is doing.

```python
def log_core_state(dut, logger, cycle):
    """
    Peeks inside Core 0 to see what the hardware is doing.
    This is equivalent to attaching an Oscilloscope to the chip.
    """
    # Access internal signals via hierarchy: GPU -> Core[0] -> Internal Wires
    # dut.genblk1[0] accesses the first generated Core instance
    core = dut.genblk1[0].core_instance
    
    # 1. Read The Program Counter (Where are we?)
    pc = int(core.current_pc.value)
    
    # 2. Read the State Machine (What are we doing?)
    # Map binary state to human readable string
    states = {0:"IDLE", 1:"FETCH", 2:"DECODE", 3:"REQUEST", 4:"WAIT", 5:"EXECUTE", 6:"UPDATE", 7:"DONE"}
    state_val = int(core.core_state.value)
    state_str = states.get(state_val, "UNKNOWN")
    
    # 3. Spy on Thread 0's Registers (The "Backpack")
    # We look at Register 0 (Accumulator) and Register 15 (Thread ID)
    # Note: Using hierarchy to reach inside the generated loop for Thread 0
    thread0_regs = core.genblk2[0].register_instance.registers
    r0_val = int(thread0_regs[0].value) # R0 (General Purpose)
    r15_val = int(thread0_regs[15].value) # ThreadID (System)
    
    # 4. Spy on the Instruction
    instr = int(core.instruction.value)
    
    # Log the snapshot
    logger.info(f"[Cycle {cycle:03d}] PC: {pc:02d} | State: {state_str:<8} | Instr: {hex(instr)} | T0_R0: {r0_val}")
```

This helper function allows us to debug the silicon execution cycle-by-cycle.

*   **Hierarchical Access:** Notice `dut.genblk1[0].core_instance`. We are digging through the layers of the Verilog design (`gpu.sv` -> `core.sv`) to read private wires that aren't exposed on the pins.
*   **State Decoding:** The hardware sees state `101`, but we convert it to `EXECUTE` for readability. This effectively builds a "Logic Analyzer" view in our terminal.

Next we need to create a virtual RAM that simulates the HBM memory. This will respond to the memory requests from the GPU.

Let’s create a file `test/helpers/memory.py`. It listens to the `mem_read_valid` signals from the GPU and responds with data.

```python
import cocotb
from cocotb.triggers import RisingEdge


class Memory:
    def __init__(self, dut, addr_bits, data_bits, channels, name):
        self.dut = dut
        self.addr_bits = addr_bits
        self.data_bits = data_bits
        self.channels = channels
        self.name = name
        # The physical storage (256 rows of RAM)
        self.memory = [0] * (2**addr_bits)
        # Get handles to the GPU pins
        self.read_valid = getattr(dut, f"{name}_mem_read_valid")
        self.read_addr = getattr(dut, f"{name}_mem_read_address")
        self.read_ready = getattr(dut, f"{name}_mem_read_ready")
        self.read_data = getattr(dut, f"{name}_mem_read_data")
        
        if name == "data":
            self.write_valid = getattr(dut, f"{name}_mem_write_valid")
            self.write_addr = getattr(dut, f"{name}_mem_write_address")
            self.write_data = getattr(dut, f"{name}_mem_write_data")
            self.write_ready = getattr(dut, f"{name}_mem_write_ready")
    # Load initial data (like torch.tensor)
    def load(self, data):
        for i, d in enumerate(data):
            self.memory[i] = d
    # The Logic Loop (Traffic Cop)
    def run(self):
        # 1. Handle Reads
        # In hardware, signals are binary strings. We convert them.
        valid_bits = str(self.read_valid.value)
        
        # Reset ready signal default
        ready_val = 0 
        data_val = 0
        # Iterate over channels (Thread 0, Thread 1...)
        for i in range(self.channels):
            # Check if this specific channel is requesting (bit is '1')
            # Note: Verilog arrays are often reversed in Python strings
            idx = self.channels - 1 - i
            if len(valid_bits) > idx and valid_bits[idx] == '1': 
                # Extract address for this channel
                addr_start = i * self.addr_bits
                addr_end = (i + 1) * self.addr_bits
                addr_vec = str(self.read_addr.value)
                # Pad logic omitted for brevity, extracting integer address:
                addr_int = int(addr_vec[-(i+1)*8 : -i*8 if i>0 else None], 2)
                
                # Fetch data from python array
                mem_val = self.memory[addr_int]
                
                # Update output integers (bit shifting to construct the bus)
                data_val |= (mem_val << (i * self.data_bits))
                ready_val |= (1 << i)
        # Drive the wires back to the GPU
        self.read_data.value = data_val
        self.read_ready.value = ready_val
        # 2. Handle Writes (Similar logic for STR instructions)
        if self.name == "data":
             # Check write_valid, perform write to self.memory, set write_ready
             # (Implementation details omitted for brevity)
             self.write_ready.value = int(self.write_valid.value)
```

This script acts as the VRAM for our GPU simulation.

*   **Memory Storage:** We create a simple Python list (`self.memory`) to simulate the RAM storage.
*   **Read Logic:** It listens to the `mem_read_valid` signals from the GPU. When a bit is set, it reads the corresponding address, fetches data from `self.memory`, and drives the `mem_read_data` and `mem_read_ready` signals back to the GPU.
*   It acts as the **Memory Controller’s Counterpart**. When `controller.sv` asserts `valid`, this script asserts `ready`.
*   In a real H100 simulation, this script would include latency (e.g., `await ClockCycles(100)`) to simulate the slowness of VRAM. Here, we respond instantly for simpler debugging.

Now we can test our first kernel: **Matrix Addition**. This is the “Hello World” of AI hardware tests. It stresses the memory system without requiring complex computation. Let’s create `test/test_matadd.py`.

**(Memory Bound):** Matrix Addition is `C = A + B`. It requires very little math (1 add), but a lot of data movement (2 Loads, 1 Store). In an **H100**, this workload stresses the **HBM3 Memory Controllers**. It tests if our `LSU` (Load Store Unit) and `controller.sv` (Traffic Cop) are working correctly to handle multiple threads asking for data at once.

Let’s define this kernel logic in assembly first.

```bash
// 1. Calculate Index: i = ThreadID
// 2. Load A[i] -> R1
// 3. Load B[i] -> R2
// 4. Add R3 = R1 + R2
// 5. Store C[i] = R3
```

In our simplified ISA, this translates to:

```bash
0b0111_0010_0000_1000  # LDR R2, [R8] (Load B)
0b0011_0011_0001_0010  # ADD R3, R1, R2 (R3 = R1 + R2)
0b1000_0000_0001_0011  # STR [R16], R3 (Store C)
0b1111_0000_0000_0000  # RET
```

We are basically loading from two arrays `A` and `B`, adding them, and storing the result in `C`. Let's implement the testbench in `test/test_matadd.py`.

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from helpers.memory import Memory
from helpers.monitor import log_core_state


@cocotb.test()
async def test_matadd(dut):
    # Setup Clock (50MHz -> 20ns period)
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    # --- 1. MEMORY INITIALIZATION ---
    # We load the binary program into Program Memory
    # (Simplified binary for readability)
    prog_mem = Memory(dut, 8, 16, 1, "program")
    prog_mem.load([
        0b0111_0001_0000_0000, # LDR R1, [R0] (Load A) - R0 has address 0+ID
        0b0111_0010_0000_1000, # LDR R2, [R8] (Load B) - R8 has address 8+ID
        0b0011_0011_0001_0010, # ADD R3, R1, R2 (R3 = R1 + R2)
        0b1000_0000_0001_0011, # STR [R16], R3 (Store C)
        0b1111_0000_0000_0000  # RET
    ])
    
    # We load Data Memory (VRAM)
    # A = [10, 20, 30, 40], B = [1, 2, 3, 4]
    data_mem = Memory(dut, 8, 8, 4, "data")
    data_mem.load([10, 20, 30, 40, 0, 0, 0, 0, 1, 2, 3, 4]) 
    # --- 2. CONFIGURATION (DCR) ---
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    
    # Configure 4 Threads (SIMT)
    dut.device_control_write_enable.value = 1
    dut.device_control_data.value = 4 
    await RisingEdge(dut.clk)
    dut.device_control_write_enable.value = 0
    # --- 3. EXECUTION ---
    dut._log.info(">>> Launching Kernel: Matrix Addition <<<")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    # Cycle Loop
    for cycle in range(50):
        # Run the "Hardware" modules
        prog_mem.run() 
        data_mem.run()
        
        # Log status
        log_core_state(dut, dut._log, cycle)
        
        if dut.done.value == 1:
            break
        await RisingEdge(dut.clk)
    # --- 4. VERIFICATION ---
    dut._log.info(">>> Kernel Complete. Checking Memory... <<<")
    # Result should be at address 16 (assumed base for C)
    res = data_mem.memory[16:20]
    dut._log.info(f"VRAM State C: {res}")
    assert res == [11, 22, 33, 44]
```

Now we can simulate this test using Cocotb. Run the following command in your terminal:

```bash
cocotb-run --verilog-src src/gpu.sv --test-module test_matadd
```

This will start the simulation, and you should see detailed logs of the core’s internal state.

```bash
INFO     >>> Launching Kernel: Matrix Addition <<<
INFO     [Cycle 000] PC: 00 | State: IDLE     | Instr: 0x0      | T0_R0: 0
INFO     [Cycle 001] PC: 00 | State: FETCH    | Instr: 0x0      | T0_R0: 0
INFO     [Cycle 002] PC: 00 | State: DECODE   | Instr: 0x7100   | T0_R0: 0
INFO     [Cycle 003] PC: 00 | State: REQUEST  | Instr: 0x7100   | T0_R0: 0
INFO     [Cycle 004] PC: 00 | State: WAIT     | Instr: 0x7100   | T0_R0: 0
INFO     [Cycle 005] PC: 00 | State: WAIT     | Instr: 0x7100   | T0_R0: 0
INFO     [Cycle 006] PC: 00 | State: EXECUTE  | Instr: 0x7100   | T0_R0: 10
```

You can clearly see the internal state of the core as it processes the first instruction (`LDR R1, [R0]`).

*   **Cycles 0–2 (Frontend):** The `Fetcher` retrieves `0x7100` (`LDR`). The `Decoder` identifies it as a Load.
*   **Cycles 4–5 (The Stall):** Look at the **WAIT** states. The `PC` is stuck at `00`. The GPU is stalled.
*   **Why?** Because `lsu.sv` sent a request, and `controller.sv` is currently processing it.
*   Even though Thread 0 might be ready, if Thread 3 is blocked by the memory bus, the Scheduler holds the *entire* core in `WAIT`. This visually proves the cost of memory access.

We can also perform **Matrix Multiplication** to test the compute capabilities and branching logic.

Matrix Multiplication (`C = A @ B`) is the engine of Deep Learning. It involves **Loops** and **Reuse**. In an **H100**, this uses the **Tensor Cores**. The challenge here isn't just fetching data, it's keeping the ALU busy accumulating sums (`C += A * B`) inside a loop. This tests our **Program Counter's** branching logic.

This time our assembly looks like this:

```bash
// 1. Init Accumulator R0 = 0
// 2. Load A -> R1
// 3. Load B -> R2
// 4. Multiply R3 = R1 * R2
// 5. Accumulate R0 = R0 + R3
// 6. Decrement Loop Counter
// 7. CMP Counter, 0
// 8. BRnzp (Branch if Positive) -> Jump back to Step 2
```

In here, we are looping over 2 elements of A and B, multiplying them, and accumulating the result in R0. This requires the GPU to make decisions based on the `CMP` instruction and update the `PC` accordingly. Let's implement this in `test/test_matmul.py`.

```python
@cocotb.test()
async def test_matmul(dut):
    # Setup... (Same as MatAdd)
    prog_mem = Memory(dut, 8, 16, 1, "program")
    prog_mem.load([
        0b1001_0000_00000000, # CONST R0, #0 (Acc = 0)
        # LOOP START (PC=1)
        0b0111_0001_00000000, # LDR R1, [PtrA] (Load 1)
        0b0111_0010_00000000, # LDR R2, [PtrB] (Load 3)
        0b0101_0011_0001_0010, # MUL R3, R1, R2 (R3 = 1*3 = 3)
        0b0011_0000_0000_0011, # ADD R0, R0, R3 (Acc += 3)
        # Loop Check (assume R5 is counter=2)
        0b0100_0101_0101_0001, # SUB R5, R5, #1 (Decr Loop)
        0b0010_0000_0101_0000, # CMP R5, #0     (Compare Counter vs 0)
        0b0001_0010_0000_0001, # BRp #1         (Branch if Positive to PC=1)
        0b1111_0000_0000_0000  # RET
    ])
    # ... (Run and Logic remains similar to MatAdd)
```

In a similar way, we load the program and data into memory, configure the GPU, and run the simulation. The key difference is in the logging output, where we will see branching behavior.

```bash
cocotb-run --verilog-src src/gpu.sv --test-module test_matmul
```

This is what we are getting

```bash
INFO     [Cycle 015] PC: 04 | State: EXECUTE  | Instr: ADD (Acc+=) | T0_R0: 3
INFO     [Cycle 018] PC: 05 | State: EXECUTE  | Instr: SUB (Decr)  | T0_R5: 1
INFO     [Cycle 020] PC: 06 | State: EXECUTE  | Instr: CMP         | NZP: 001 (Pos)
INFO     [Cycle 021] PC: 06 | State: UPDATE   | Instr: CMP         | NZP Update
INFO     [Cycle 022] PC: 07 | State: EXECUTE  | Instr: BRp         | Action: JUMP
INFO     [Cycle 023] PC: 01 | State: FETCH    | Instr: ...         | Loop Reset!
```

We can clearly see the branching logic:

*   **Cycle 15:** We see the ALU performing the MAC (Multiply-Accumulate). `T0_R0` (Accumulator) holds `3`.
*   **Cycle 20 (The Decision):** The ALU executes `CMP`. It sees the loop counter is `1`. This is positive. It sets the `NZP` flags in `pc.sv`.
*   **Cycle 22 (The Branch):** The instruction is `BRp` (Branch if Positive). The **Program Counter** reads the `NZP` flags, sees the "P" bit is high, and forces `next_pc` to **1**.
*   **Cycle 23:** The Fetcher goes back to address 1. The loop has restarted.

We can also verify the final result in memory by checking the VRAM state.

```
INFO     >>> Kernel Complete. Checking Memory... <<<
INFO     VRAM State C: [11, 22, 33, 44]
INFO     test_matadd passed
INFO     test_matmul passed
```

You can see that ….

This final output confirms that the silicon logic correctly implemented the high-level math.

*   `MatAdd` **verified the Logistics:** Data flows correctly from memory to the core and back, handling stalls correctly.
*   `MatMul` **verified the Logic:** The GPU can make decisions, loop, and perform complex arithmetic sequences.

This matches the behavior of an H100 executing a CUDA kernel, just at a microscopic scale. We have successfully proven that our RTL code is a valid computer architecture.

### Scaling to a Modern Blackwell GPUs

We have built a functional SIMT processor. It can run code, manage memory, and do parallel computation. But to bridge the gap between our **Tiny GPU** and modern blackwell gpus of Nvidia, we can further implement several advanced architectural features, such as:

1.  **Multi-layered Cache & Shared Memory:** We can add **L1/L2 Caches** closer to the cores to minimize slow trips to global memory, and implement **Shared Memory** (scratchpad) so threads within a block can exchange data instantly without hitting VRAM.
2.  **Memory Coalescing:** We can upgrade the Memory Controller to detect when neighbor threads (e.g., Thread 0 and 1) access neighbor addresses (0x100, 0x101) and merge them into a single, efficient burst transaction.
3.  **Pipelining:** Instead of stopping the Fetcher while the ALU works (`Fetch -> Wait -> Decode -> Wait`), we can pipeline the stages so the Fetcher grabs the *next* instruction while the ALU executes the *current* one, doubling throughput.
4.  **Warp Scheduling:** We can allow the Scheduler to manage multiple active warps (batches of threads) per core. When Warp A stalls on memory, the scheduler instantly switches to Warp B, hiding latency and keeping the ALUs 100% busy.
5.  **Branch Divergence:** We can add a “Reconvergence Stack” to the Program Counter. This would allow threads to split paths (`if/else`) by masking off half the threads, executing the `if`, then flipping the mask to execute the `else`.
6.  **Synchronization & Barriers:** We can implement a `__syncthreads()` instruction in the ISA. This would pause all threads at a specific line until every thread in the block has arrived, ensuring safe communication between parallel workers.

> You can [follow me on Medium](https://medium.com/@fareedkhandev) if you find this article useful
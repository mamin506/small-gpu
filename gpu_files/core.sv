`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// COMPUTE CORE
// > Handles processing 1 block at a time
// > The core also has its own scheduler to manage control flow
// > Each core contains 1 fetcher & decoder, and register files, ALUs, LSUs, PC for each thread
module core #(
    parameter DATA_MEM_ADDR_BITS = 8,         // Address width for data memory
    parameter DATA_MEM_DATA_BITS = 8,         // Data width for data memory
    parameter PROGRAM_MEM_ADDR_BITS = 8,       // Address width for program memory
    parameter PROGRAM_MEM_DATA_BITS = 16,      // Data width for program memory
    parameter THREADS_PER_BLOCK = 4            // Number of threads per block
) (
    input wire clk,                            // Clock input
    input wire reset,                          // Reset input

    // Kernel Execution
    input wire start,                          // Start signal for kernel execution
    output wire done,                          // Done signal for kernel execution

    // Block Metadata
    input wire [7:0] block_id,                 // Block identifier
    input wire [$clog2(THREADS_PER_BLOCK):0] thread_count, // Number of active threads

    // Program Memory
    output reg program_mem_read_valid,         // Program memory read valid signal
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] program_mem_read_address, // Program memory read address
    input reg program_mem_read_ready,          // Program memory read ready signal
    input reg [PROGRAM_MEM_DATA_BITS-1:0] program_mem_read_data, // Program memory read data

    // Data Memory
    output reg [THREADS_PER_BLOCK-1:0] data_mem_read_valid, // Data memory read valid per thread
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_read_address [THREADS_PER_BLOCK-1:0], // Data memory read address per thread
    input reg [THREADS_PER_BLOCK-1:0] data_mem_read_ready, // Data memory read ready per thread
    input reg [DATA_MEM_DATA_BITS-1:0] data_mem_read_data [THREADS_PER_BLOCK-1:0], // Data memory read data per thread
    output reg [THREADS_PER_BLOCK-1:0] data_mem_write_valid, // Data memory write valid per thread
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_write_address [THREADS_PER_BLOCK-1:0], // Data memory write address per thread
    output reg [DATA_MEM_DATA_BITS-1:0] data_mem_write_data [THREADS_PER_BLOCK-1:0], // Data memory write data per thread
    input reg [THREADS_PER_BLOCK-1:0] data_mem_write_ready // Data memory write ready per thread
);
    // State
    reg [2:0] core_state;                      // Core state register
    reg [2:0] fetcher_state;                   // Fetcher state register
    reg [15:0] instruction;                    // Current instruction register

    // Intermediate Signals
    reg [7:0] current_pc;                      // Current program counter
    wire [7:0] next_pc[THREADS_PER_BLOCK-1:0]; // Next program counter for each thread
    reg [7:0] rs[THREADS_PER_BLOCK-1:0];       // Source register value for each thread
    reg [7:0] rt[THREADS_PER_BLOCK-1:0];       // Target register value for each thread
    reg [1:0] lsu_state[THREADS_PER_BLOCK-1:0];// LSU state for each thread
    reg [7:0] lsu_out[THREADS_PER_BLOCK-1:0];  // LSU output for each thread
    wire [7:0] alu_out[THREADS_PER_BLOCK-1:0]; // ALU output for each thread
    
    // Decoded Instruction Signals
    reg [3:0] decoded_rd_address;              // Decoded destination register address
    reg [3:0] decoded_rs_address;              // Decoded source register address
    reg [3:0] decoded_rt_address;              // Decoded target register address
    reg [2:0] decoded_nzp;                     // Decoded NZP flags
    reg [7:0] decoded_immediate;               // Decoded immediate value

    // Decoded Control Signals
    reg decoded_reg_write_enable;              // Enable writing to a register
    reg decoded_mem_read_enable;               // Enable reading from memory
    reg decoded_mem_write_enable;              // Enable writing to memory
    reg decoded_nzp_write_enable;              // Enable writing to NZP register
    reg [1:0] decoded_reg_input_mux;           // Select input to register
    reg [1:0] decoded_alu_arithmetic_mux;      // Select arithmetic operation
    reg decoded_alu_output_mux;                // Select operation in ALU
    reg decoded_pc_mux;                        // Select source of next PC
    reg decoded_ret;                           // Return instruction flag

    // Fetcher
    fetcher #(
        .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS), // Program memory address width
        .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS)  // Program memory data width
    ) fetcher_instance (
        .clk(clk),                                   // Clock input
        .reset(reset),                               // Reset input
        .core_state(core_state),                     // Core state input
        .current_pc(current_pc),                     // Current PC input
        .mem_read_valid(program_mem_read_valid),     // Program memory read valid output
        .mem_read_address(program_mem_read_address), // Program memory read address output
        .mem_read_ready(program_mem_read_ready),     // Program memory read ready input
        .mem_read_data(program_mem_read_data),       // Program memory read data input
        .fetcher_state(fetcher_state),               // Fetcher state output
        .instruction(instruction)                    // Instruction output
    );

    // Decoder
    decoder decoder_instance (
        .clk(clk),                                   // Clock input
        .reset(reset),                               // Reset input
        .core_state(core_state),                     // Core state input
        .instruction(instruction),                   // Instruction input
        .decoded_rd_address(decoded_rd_address),     // Decoded destination register address output
        .decoded_rs_address(decoded_rs_address),     // Decoded source register address output
        .decoded_rt_address(decoded_rt_address),     // Decoded target register address output
        .decoded_nzp(decoded_nzp),                   // Decoded NZP flags output
        .decoded_immediate(decoded_immediate),       // Decoded immediate value output
        .decoded_reg_write_enable(decoded_reg_write_enable), // Register write enable output
        .decoded_mem_read_enable(decoded_mem_read_enable),   // Memory read enable output
        .decoded_mem_write_enable(decoded_mem_write_enable), // Memory write enable output
        .decoded_nzp_write_enable(decoded_nzp_write_enable), // NZP write enable output
        .decoded_reg_input_mux(decoded_reg_input_mux),       // Register input mux output
        .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux), // ALU arithmetic mux output
        .decoded_alu_output_mux(decoded_alu_output_mux),     // ALU output mux output
        .decoded_pc_mux(decoded_pc_mux),                     // PC mux output
        .decoded_ret(decoded_ret)                            // Return flag output
    );

    // Scheduler
    scheduler #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK), // Number of threads per block
    ) scheduler_instance (
        .clk(clk),                                   // Clock input
        .reset(reset),                               // Reset input
        .start(start),                               // Start signal input
        .fetcher_state(fetcher_state),               // Fetcher state input
        .core_state(core_state),                     // Core state output
        .decoded_mem_read_enable(decoded_mem_read_enable),   // Memory read enable input
        .decoded_mem_write_enable(decoded_mem_write_enable), // Memory write enable input
        .decoded_ret(decoded_ret),                   // Return flag input
        .lsu_state(lsu_state),                       // LSU state input
        .current_pc(current_pc),                     // Current PC input/output
        .next_pc(next_pc),                           // Next PC output
        .done(done)                                  // Done signal output
    );

    // Dedicated ALU, LSU, registers, & PC unit for each thread this core has capacity for
    genvar i;                                        // Generate variable for loop
    generate
        for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads // Loop for each thread
            // ALU
            alu alu_instance (
                .clk(clk),                           // Clock input
                .reset(reset),                       // Reset input
                .enable(i < thread_count),           // Enable if thread is active
                .core_state(core_state),             // Core state input
                .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux), // ALU arithmetic mux input
                .decoded_alu_output_mux(decoded_alu_output_mux),         // ALU output mux input
                .rs(rs[i]),                          // Source register value input
                .rt(rt[i]),                          // Target register value input
                .alu_out(alu_out[i])                 // ALU output
            );

            // LSU
            lsu lsu_instance (
                .clk(clk),                           // Clock input
                .reset(reset),                       // Reset input
                .enable(i < thread_count),           // Enable if thread is active
                .core_state(core_state),             // Core state input
                .decoded_mem_read_enable(decoded_mem_read_enable),   // Memory read enable input
                .decoded_mem_write_enable(decoded_mem_write_enable), // Memory write enable input
                .mem_read_valid(data_mem_read_valid[i]),             // Data memory read valid output
                .mem_read_address(data_mem_read_address[i]),         // Data memory read address output
                .mem_read_ready(data_mem_read_ready[i]),             // Data memory read ready input
                .mem_read_data(data_mem_read_data[i]),               // Data memory read data input
                .mem_write_valid(data_mem_write_valid[i]),           // Data memory write valid output
                .mem_write_address(data_mem_write_address[i]),       // Data memory write address output
                .mem_write_data(data_mem_write_data[i]),             // Data memory write data output
                .mem_write_ready(data_mem_write_ready[i]),           // Data memory write ready input
                .rs(rs[i]),                          // Source register value input
                .rt(rt[i]),                          // Target register value input
                .lsu_state(lsu_state[i]),            // LSU state output
                .lsu_out(lsu_out[i])                 // LSU output
            );

            // Register File
            registers #(
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK), // Number of threads per block
                .THREAD_ID(i),                         // Thread ID
                .DATA_BITS(DATA_MEM_DATA_BITS),        // Data width
            ) register_instance (
                .clk(clk),                           // Clock input
                .reset(reset),                       // Reset input
                .enable(i < thread_count),           // Enable if thread is active
                .block_id(block_id),                 // Block ID input
                .core_state(core_state),             // Core state input
                .decoded_reg_write_enable(decoded_reg_write_enable), // Register write enable input
                .decoded_reg_input_mux(decoded_reg_input_mux),       // Register input mux input
                .decoded_rd_address(decoded_rd_address),             // Destination register address input
                .decoded_rs_address(decoded_rs_address),             // Source register address input
                .decoded_rt_address(decoded_rt_address),             // Target register address input
                .decoded_immediate(decoded_immediate),               // Immediate value input
                .alu_out(alu_out[i]),                // ALU output input
                .lsu_out(lsu_out[i]),                // LSU output input
                .rs(rs[i]),                          // Source register value output
                .rt(rt[i])                           // Target register value output
            );

            // Program Counter
            pc #(
                .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS), // Data memory data width
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS) // Program memory address width
            ) pc_instance (
                .clk(clk),                           // Clock input
                .reset(reset),                       // Reset input
                .enable(i < thread_count),           // Enable if thread is active
                .core_state(core_state),             // Core state input
                .decoded_nzp(decoded_nzp),           // Decoded NZP flags input
                .decoded_immediate(decoded_immediate), // Immediate value input
                .decoded_nzp_write_enable(decoded_nzp_write_enable), // NZP write enable input
                .decoded_pc_mux(decoded_pc_mux),     // PC mux input
                .alu_out(alu_out[i]),                // ALU output input
                .current_pc(current_pc),             // Current PC input/output
                .next_pc(next_pc[i])                 // Next PC output
            );
        end
    endgenerate
endmodule // End of core module

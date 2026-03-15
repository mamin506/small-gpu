`default_nettype none
`timescale 1ns/1ns

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
    localparam ARITHMETIC = 2'b00, // Data coming from ALU (ADD, SUB)
        MEMORY = 2'b01,            // Data coming from RAM (LDR)
        CONSTANT = 2'b10;          // Data is a hardcoded constant (CONST)

    // 16 registers per thread (13 free registers and 3 read-only registers)
    reg [7:0] registers[15:0];

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
    end
endmodule

`default_nettype none
`timescale 1ns/1ns

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

    localparam IDLE = 3'b000, // Waiting to start
		FETCH = 3'b001,       // Fetch instructions from program memory
		DECODE = 3'b010,      // Decode instructions into control signals
		REQUEST = 3'b011,     // Request data from registers or memory
		WAIT = 3'b100,        // Wait for response from memory if necessary
		EXECUTE = 3'b101,     // Execute ALU and PC calculations
		UPDATE = 3'b110,      // Update registers, NZP, and PC
		DONE = 3'b111;        // Done executing this block

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
        end
    end
endmodule

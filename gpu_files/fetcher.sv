`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// INSTRUCTION FETCHER
// > Retrieves the instruction at the current PC from global data memory
// > Each core has its own fetcher
module fetcher #(
    parameter PROGRAM_MEM_ADDR_BITS = 8,   // Number of bits for program memory address
    parameter PROGRAM_MEM_DATA_BITS = 16   // Number of bits for program memory data
) (
    input wire clk,                        // Clock input
    input wire reset,                      // Reset input (active high)
    
    // Execution State
    input reg [2:0] core_state,            // Current state of the core (3 bits)
    input reg [7:0] current_pc,            // Current program counter (8 bits)

    // Program Memory
    output reg mem_read_valid,                                 // Assert to request memory read
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] mem_read_address,   // Address to read from program memory
    input reg mem_read_ready,                                  // Indicates memory read data is ready
    input reg [PROGRAM_MEM_DATA_BITS-1:0] mem_read_data,       // Data read from program memory

    // Fetcher Output
    output reg [2:0] fetcher_state,                            // Current state of the fetcher
    output reg [PROGRAM_MEM_DATA_BITS-1:0] instruction,        // Output instruction fetched
);
    // State encoding for fetcher FSM
    localparam IDLE = 3'b000,      // Idle state
        FETCHING = 3'b001,         // Fetching instruction from memory
        FETCHED = 3'b010;          // Instruction fetched and ready

    // Sequential logic: runs on rising edge of clock
    always @(posedge clk) begin
        if (reset) begin
            fetcher_state <= IDLE;                                 // Reset fetcher state to IDLE
            mem_read_valid <= 0;                                   // Deassert memory read valid
            mem_read_address <= 0;                                 // Clear memory read address
            instruction <= {PROGRAM_MEM_DATA_BITS{1'b0}};          // Clear instruction output
        end else begin
            case (fetcher_state)
                IDLE: begin
                    // Start fetching when core_state = FETCH (3'b001)
                    if (core_state == 3'b001) begin
                        fetcher_state <= FETCHING;                 // Move to FETCHING state
                        mem_read_valid <= 1;                       // Assert memory read valid
                        mem_read_address <= current_pc;            // Set memory read address to current PC
                    end
                end
                FETCHING: begin
                    // Wait for response from program memory
                    if (mem_read_ready) begin
                        fetcher_state <= FETCHED;                  // Move to FETCHED state
                        instruction <= mem_read_data;              // Store the instruction received
                        mem_read_valid <= 0;                       // Deassert memory read valid
                    end
                end
                FETCHED: begin
                    // Reset when core_state = DECODE (3'b010)
                    if (core_state == 3'b010) begin 
                        fetcher_state <= IDLE;                     // Return to IDLE state
                    end
                end
            endcase
        end
    end
endmodule

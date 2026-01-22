`default_nettype none // Disable implicit net declarations
`timescale 1ns/1ns    // Set simulation time unit and precision

// LOAD-STORE UNIT
// > Handles asynchronous memory load and store operations and waits for response
// > Each thread in each core has its own LSU
// > LDR, STR instructions are executed here
module lsu (
    input wire clk,                        // Clock signal
    input wire reset,                      // Reset signal (active high)
    input wire enable,                     // Enable signal for this LSU instance

    // State
    input reg [2:0] core_state,            // Current state of the core

    // Memory Control Signals
    input reg decoded_mem_read_enable,     // Indicates if a memory read is requested
    input reg decoded_mem_write_enable,    // Indicates if a memory write is requested

    // Registers
    input reg [7:0] rs,                    // Source register/address for memory operation
    input reg [7:0] rt,                    // Data to write (for store)

    // Data Memory
    output reg mem_read_valid,             // Assert to request a memory read
    output reg [7:0] mem_read_address,     // Address to read from
    input reg mem_read_ready,              // Memory signals read data is ready
    input reg [7:0] mem_read_data,         // Data returned from memory read
    output reg mem_write_valid,            // Assert to request a memory write
    output reg [7:0] mem_write_address,    // Address to write to
    output reg [7:0] mem_write_data,       // Data to write to memory
    input reg mem_write_ready,             // Memory signals write is complete

    // LSU Outputs
    output reg [1:0] lsu_state,            // Current state of the LSU
    output reg [7:0] lsu_out               // Output data from LSU (for loads)
);
    // State encoding for LSU state machine
    localparam IDLE = 2'b00,               // Idle state
               REQUESTING = 2'b01,         // Requesting memory operation
               WAITING = 2'b10,            // Waiting for memory response
               DONE = 2'b11;               // Operation complete

    // Sequential logic: state machine for LSU
    always @(posedge clk) begin
        if (reset) begin
            lsu_state <= IDLE;             // Reset LSU state to IDLE
            lsu_out <= 0;                  // Clear output data
            mem_read_valid <= 0;           // Deassert memory read valid
            mem_read_address <= 0;         // Clear memory read address
            mem_write_valid <= 0;          // Deassert memory write valid
            mem_write_address <= 0;        // Clear memory write address
            mem_write_data <= 0;           // Clear memory write data
        end else if (enable) begin
            // If memory read enable is triggered (LDR instruction)
            if (decoded_mem_read_enable) begin 
                case (lsu_state)
                    IDLE: begin
                        // Only proceed if core_state is REQUEST
                        if (core_state == 3'b011) begin 
                            lsu_state <= REQUESTING; // Move to REQUESTING state
                        end
                    end
                    REQUESTING: begin 
                        mem_read_valid <= 1;        // Assert memory read valid
                        mem_read_address <= rs;     // Set memory read address
                        lsu_state <= WAITING;       // Move to WAITING state
                    end
                    WAITING: begin
                        if (mem_read_ready == 1) begin
                            mem_read_valid <= 0;    // Deassert memory read valid
                            lsu_out <= mem_read_data; // Capture read data
                            lsu_state <= DONE;      // Move to DONE state
                        end
                    end
                    DONE: begin 
                        // Reset when core_state is UPDATE
                        if (core_state == 3'b110) begin 
                            lsu_state <= IDLE;      // Return to IDLE state
                        end
                    end
                endcase
            end

            // If memory write enable is triggered (STR instruction)
            if (decoded_mem_write_enable) begin 
                case (lsu_state)
                    IDLE: begin
                        // Only proceed if core_state is REQUEST
                        if (core_state == 3'b011) begin 
                            lsu_state <= REQUESTING; // Move to REQUESTING state
                        end
                    end
                    REQUESTING: begin 
                        mem_write_valid <= 1;       // Assert memory write valid
                        mem_write_address <= rs;    // Set memory write address
                        mem_write_data <= rt;       // Set memory write data
                        lsu_state <= WAITING;       // Move to WAITING state
                    end
                    WAITING: begin
                        if (mem_write_ready) begin
                            mem_write_valid <= 0;   // Deassert memory write valid
                            lsu_state <= DONE;      // Move to DONE state
                        end
                    end
                    DONE: begin 
                        // Reset when core_state is UPDATE
                        if (core_state == 3'b110) begin 
                            lsu_state <= IDLE;      // Return to IDLE state
                        end
                    end
                endcase
            end
        end
    end
endmodule

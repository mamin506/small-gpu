`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// BLOCK DISPATCH
// > The GPU has one dispatch unit at the top level
// > Manages processing of threads and marks kernel execution as done
// > Sends off batches of threads in blocks to be executed by available compute cores
module dispatch #(
    parameter NUM_CORES = 2,              // Number of compute cores
    parameter THREADS_PER_BLOCK = 4       // Number of threads per block
) (
    input wire clk,                       // Clock signal
    input wire reset,                     // Reset signal (active high)
    input wire start,                     // Start signal to begin kernel execution

    // Kernel Metadata
    input wire [7:0] thread_count,        // Total number of threads to execute

    // Core States
    input reg [NUM_CORES-1:0] core_done,  // Indicates which cores have finished their block
    output reg [NUM_CORES-1:0] core_start, // Signal to start a block on each core
    output reg [NUM_CORES-1:0] core_reset, // Signal to reset each core
    output reg [7:0] core_block_id [NUM_CORES-1:0], // Block ID assigned to each core
    output reg [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0], // Threads assigned to each core

    // Kernel Execution
    output reg done                       // Indicates kernel execution is complete
);
    // Calculate the total number of blocks based on total threads & threads per block
    wire [7:0] total_blocks;              // Total number of blocks to dispatch
    assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // Ceiling division

    // Keep track of how many blocks have been processed
    reg [7:0] blocks_dispatched;          // Number of blocks dispatched to cores
    reg [7:0] blocks_done;                // Number of blocks finished processing
    reg start_execution;                  // Internal flag to indicate execution has started

    always @(posedge clk) begin           // On every rising edge of the clock
        if (reset) begin                  // If reset is asserted
            done <= 0;                    // Clear the done flag
            blocks_dispatched = 0;        // Reset blocks dispatched counter
            blocks_done = 0;              // Reset blocks done counter
            start_execution <= 0;         // Clear the start execution flag

            for (int i = 0; i < NUM_CORES; i++) begin // For each core
                core_start[i] <= 0;       // Deassert core start signal
                core_reset[i] <= 1;       // Assert core reset signal
                core_block_id[i] <= 0;    // Clear block ID for the core
                core_thread_count[i] <= THREADS_PER_BLOCK; // Set thread count to max per block
            end
        end else if (start) begin         // If start signal is asserted
            // EDA: Indirect way to get @(posedge start) without driving from 2 different clocks
            if (!start_execution) begin   // If execution hasn't started yet
                start_execution <= 1;     // Set execution started flag
                for (int i = 0; i < NUM_CORES; i++) begin // For each core
                    core_reset[i] <= 1;   // Assert core reset signal
                end
            end

            // If the last block has finished processing, mark this kernel as done executing
            if (blocks_done == total_blocks) begin // If all blocks are done
                done <= 1;                // Set done flag
            end

            for (int i = 0; i < NUM_CORES; i++) begin // For each core
                if (core_reset[i]) begin  // If core is being reset
                    core_reset[i] <= 0;   // Deassert core reset signal

                    // If this core was just reset, check if there are more blocks to be dispatched
                    if (blocks_dispatched < total_blocks) begin // If blocks remain to dispatch
                        core_start[i] <= 1; // Assert core start signal
                        core_block_id[i] <= blocks_dispatched; // Assign block ID to core
                        core_thread_count[i] <= (blocks_dispatched == total_blocks - 1) 
                            ? thread_count - (blocks_dispatched * THREADS_PER_BLOCK) // For last block, assign remaining threads
                            : THREADS_PER_BLOCK; // Otherwise, assign max threads per block

                        blocks_dispatched = blocks_dispatched + 1; // Increment blocks dispatched
                    end
                end
            end

            for (int i = 0; i < NUM_CORES; i++) begin // For each core
                if (core_start[i] && core_done[i]) begin // If core started and is now done
                    // If a core just finished executing its current block, reset it
                    core_reset[i] <= 1;   // Assert core reset signal
                    core_start[i] <= 0;   // Deassert core start signal
                    blocks_done = blocks_done + 1; // Increment blocks done
                end
            end
        end
    end
endmodule
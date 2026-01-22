`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// MEMORY CONTROLLER
// > Receives memory requests from all cores
// > Throttles requests based on limited external memory bandwidth
// > Waits for responses from external memory and distributes them back to cores
module controller #(
    parameter ADDR_BITS = 8,         // Number of address bits
    parameter DATA_BITS = 16,        // Number of data bits
    parameter NUM_CONSUMERS = 4,     // Number of consumers accessing memory through this controller
    parameter NUM_CHANNELS = 1,      // Number of concurrent channels available to send requests to global memory
    parameter WRITE_ENABLE = 1       // Whether this memory controller can write to memory (program memory is read-only)
) (
    input wire clk,                  // Clock signal
    input wire reset,                // Reset signal

    // Consumer Interface (Fetchers / LSUs)
    input reg [NUM_CONSUMERS-1:0] consumer_read_valid,                  // Read request valid from each consumer
    input reg [ADDR_BITS-1:0] consumer_read_address [NUM_CONSUMERS-1:0],// Read address from each consumer
    output reg [NUM_CONSUMERS-1:0] consumer_read_ready,                 // Read ready signal to each consumer
    output reg [DATA_BITS-1:0] consumer_read_data [NUM_CONSUMERS-1:0],  // Read data to each consumer
    input reg [NUM_CONSUMERS-1:0] consumer_write_valid,                 // Write request valid from each consumer
    input reg [ADDR_BITS-1:0] consumer_write_address [NUM_CONSUMERS-1:0],// Write address from each consumer
    input reg [DATA_BITS-1:0] consumer_write_data [NUM_CONSUMERS-1:0],  // Write data from each consumer
    output reg [NUM_CONSUMERS-1:0] consumer_write_ready,                // Write ready signal to each consumer

    // Memory Interface (Data / Program)
    output reg [NUM_CHANNELS-1:0] mem_read_valid,                       // Read request valid to memory for each channel
    output reg [ADDR_BITS-1:0] mem_read_address [NUM_CHANNELS-1:0],     // Read address to memory for each channel
    input reg [NUM_CHANNELS-1:0] mem_read_ready,                        // Read ready from memory for each channel
    input reg [DATA_BITS-1:0] mem_read_data [NUM_CHANNELS-1:0],         // Read data from memory for each channel
    output reg [NUM_CHANNELS-1:0] mem_write_valid,                      // Write request valid to memory for each channel
    output reg [ADDR_BITS-1:0] mem_write_address [NUM_CHANNELS-1:0],    // Write address to memory for each channel
    output reg [DATA_BITS-1:0] mem_write_data [NUM_CHANNELS-1:0],       // Write data to memory for each channel
    input reg [NUM_CHANNELS-1:0] mem_write_ready                        // Write ready from memory for each channel
);
    localparam IDLE = 3'b000,           // State: Idle
        READ_WAITING = 3'b010,          // State: Waiting for read response
        WRITE_WAITING = 3'b011,         // State: Waiting for write response
        READ_RELAYING = 3'b100,         // State: Relaying read data to consumer
        WRITE_RELAYING = 3'b101;        // State: Relaying write ready to consumer

    // Keep track of state for each channel and which jobs each channel is handling
    reg [2:0] controller_state [NUM_CHANNELS-1:0];                        // State of each channel
    reg [$clog2(NUM_CONSUMERS)-1:0] current_consumer [NUM_CHANNELS-1:0];  // Which consumer is each channel currently serving
    reg [NUM_CONSUMERS-1:0] channel_serving_consumer;                     // Which consumers are being served (prevents duplicate handling)

    always @(posedge clk) begin
        if (reset) begin 
            mem_read_valid <= 0;              // Reset memory read valid signals
            mem_read_address <= 0;            // Reset memory read addresses

            mem_write_valid <= 0;             // Reset memory write valid signals
            mem_write_address <= 0;           // Reset memory write addresses
            mem_write_data <= 0;              // Reset memory write data

            consumer_read_ready <= 0;         // Reset consumer read ready signals
            consumer_read_data <= 0;          // Reset consumer read data
            consumer_write_ready <= 0;        // Reset consumer write ready signals

            current_consumer <= 0;            // Reset current consumer tracking
            controller_state <= 0;            // Reset controller state

            channel_serving_consumer = 0;     // Reset channel serving consumer tracking
        end else begin 
            // For each channel, we handle processing concurrently
            for (int i = 0; i < NUM_CHANNELS; i = i + 1) begin 
                case (controller_state[i])
                    IDLE: begin
                        // While this channel is idle, cycle through consumers looking for one with a pending request
                        for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin 
                            if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin 
                                channel_serving_consumer[j] = 1;                 // Mark this consumer as being served
                                current_consumer[i] <= j;                        // Assign this consumer to the channel

                                mem_read_valid[i] <= 1;                          // Issue read request to memory
                                mem_read_address[i] <= consumer_read_address[j]; // Set memory read address
                                controller_state[i] <= READ_WAITING;             // Move to waiting for read response

                                // Once we find a pending request, pick it up with this channel and stop looking for requests
                                break;
                            end else if (consumer_write_valid[j] && !channel_serving_consumer[j]) begin 
                                channel_serving_consumer[j] = 1;                 // Mark this consumer as being served
                                current_consumer[i] <= j;                        // Assign this consumer to the channel

                                mem_write_valid[i] <= 1;                         // Issue write request to memory
                                mem_write_address[i] <= consumer_write_address[j];// Set memory write address
                                mem_write_data[i] <= consumer_write_data[j];     // Set memory write data
                                controller_state[i] <= WRITE_WAITING;            // Move to waiting for write response

                                // Once we find a pending request, pick it up with this channel and stop looking for requests
                                break;
                            end
                        end
                    end
                    READ_WAITING: begin
                        // Wait for response from memory for pending read request
                        if (mem_read_ready[i]) begin 
                            mem_read_valid[i] <= 0;                                 // Clear memory read valid
                            consumer_read_ready[current_consumer[i]] <= 1;          // Signal consumer that data is ready
                            consumer_read_data[current_consumer[i]] <= mem_read_data[i]; // Provide read data to consumer
                            controller_state[i] <= READ_RELAYING;                   // Move to relaying read data
                        end
                    end
                    WRITE_WAITING: begin 
                        // Wait for response from memory for pending write request
                        if (mem_write_ready[i]) begin 
                            mem_write_valid[i] <= 0;                                // Clear memory write valid
                            consumer_write_ready[current_consumer[i]] <= 1;         // Signal consumer that write is complete
                            controller_state[i] <= WRITE_RELAYING;                  // Move to relaying write ready
                        end
                    end
                    // Wait until consumer acknowledges it received response, then reset
                    READ_RELAYING: begin
                        if (!consumer_read_valid[current_consumer[i]]) begin 
                            channel_serving_consumer[current_consumer[i]] = 0;      // Mark consumer as no longer being served
                            consumer_read_ready[current_consumer[i]] <= 0;          // Clear consumer read ready
                            controller_state[i] <= IDLE;                            // Return channel to idle
                        end
                    end
                    WRITE_RELAYING: begin 
                        if (!consumer_write_valid[current_consumer[i]]) begin 
                            channel_serving_consumer[current_consumer[i]] = 0;      // Mark consumer as no longer being served
                            consumer_write_ready[current_consumer[i]] <= 0;         // Clear consumer write ready
                            controller_state[i] <= IDLE;                            // Return channel to idle
                        end
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// GPU
// > Built to use an external async memory with multi-channel read/write
// > Assumes that the program is loaded into program memory, data into data memory, and threads into
//   the device control register before the start signal is triggered
// > Has memory controllers to interface between external memory and its multiple cores
// > Configurable number of cores and thread capacity per core

module gpu #(
    parameter DATA_MEM_ADDR_BITS = 8,        // Number of bits in data memory address (256 rows)
    parameter DATA_MEM_DATA_BITS = 8,        // Number of bits in data memory value (8 bit data)
    parameter DATA_MEM_NUM_CHANNELS = 4,     // Number of concurrent channels for sending requests to data memory
    parameter PROGRAM_MEM_ADDR_BITS = 8,     // Number of bits in program memory address (256 rows)
    parameter PROGRAM_MEM_DATA_BITS = 16,    // Number of bits in program memory value (16 bit instruction)
    parameter PROGRAM_MEM_NUM_CHANNELS = 1,  // Number of concurrent channels for sending requests to program memory
    parameter NUM_CORES = 2,                 // Number of cores to include in this GPU
    parameter THREADS_PER_BLOCK = 4          // Number of threads to handle per block (determines the compute resources of each core)
) (
    input wire clk,                                              // Clock input
    input wire reset,                                            // Reset input

    // Kernel Execution
    input wire start,                                            // Start signal for kernel execution
    output wire done,                                            // Done signal for kernel execution

    // Device Control Register
    input wire device_control_write_enable,                      // Write enable for device control register
    input wire [7:0] device_control_data,                        // Data for device control register

    // Program Memory
    output wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_valid, // Valid signal for program memory read
    output wire [PROGRAM_MEM_ADDR_BITS-1:0] program_mem_read_address [PROGRAM_MEM_NUM_CHANNELS-1:0], // Address for program memory read
    input wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_ready,  // Ready signal for program memory read
    input wire [PROGRAM_MEM_DATA_BITS-1:0] program_mem_read_data [PROGRAM_MEM_NUM_CHANNELS-1:0],     // Data from program memory read

    // Data Memory
    output wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_read_valid, // Valid signal for data memory read
    output wire [DATA_MEM_ADDR_BITS-1:0] data_mem_read_address [DATA_MEM_NUM_CHANNELS-1:0], // Address for data memory read
    input wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_read_ready,  // Ready signal for data memory read
    input wire [DATA_MEM_DATA_BITS-1:0] data_mem_read_data [DATA_MEM_NUM_CHANNELS-1:0],     // Data from data memory read
    output wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_write_valid, // Valid signal for data memory write
    output wire [DATA_MEM_ADDR_BITS-1:0] data_mem_write_address [DATA_MEM_NUM_CHANNELS-1:0], // Address for data memory write
    output wire [DATA_MEM_DATA_BITS-1:0] data_mem_write_data [DATA_MEM_NUM_CHANNELS-1:0],    // Data for data memory write
    input wire [DATA_MEM_NUM_CHANNELS-1:0] data_mem_write_ready  // Ready signal for data memory write
);

    // Control
    wire [7:0] thread_count; // Number of threads to launch (from DCR)

    // Compute Core State
    reg [NUM_CORES-1:0] core_start; // Start signal for each core
    reg [NUM_CORES-1:0] core_reset; // Reset signal for each core
    reg [NUM_CORES-1:0] core_done;  // Done signal from each core
    reg [7:0] core_block_id [NUM_CORES-1:0]; // Block ID for each core
    reg [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0]; // Thread count for each core

    // LSU <> Data Memory Controller Channels
    localparam NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK; // Number of LSUs (one per thread)
    reg [NUM_LSUS-1:0] lsu_read_valid; // LSU read valid signals
    reg [DATA_MEM_ADDR_BITS-1:0] lsu_read_address [NUM_LSUS-1:0]; // LSU read addresses
    reg [NUM_LSUS-1:0] lsu_read_ready; // LSU read ready signals
    reg [DATA_MEM_DATA_BITS-1:0] lsu_read_data [NUM_LSUS-1:0]; // LSU read data
    reg [NUM_LSUS-1:0] lsu_write_valid; // LSU write valid signals
    reg [DATA_MEM_ADDR_BITS-1:0] lsu_write_address [NUM_LSUS-1:0]; // LSU write addresses
    reg [DATA_MEM_DATA_BITS-1:0] lsu_write_data [NUM_LSUS-1:0]; // LSU write data
    reg [NUM_LSUS-1:0] lsu_write_ready; // LSU write ready signals

    // Fetcher <> Program Memory Controller Channels
    localparam NUM_FETCHERS = NUM_CORES; // Number of fetchers (one per core)
    reg [NUM_FETCHERS-1:0] fetcher_read_valid; // Fetcher read valid signals
    reg [PROGRAM_MEM_ADDR_BITS-1:0] fetcher_read_address [NUM_FETCHERS-1:0]; // Fetcher read addresses
    reg [NUM_FETCHERS-1:0] fetcher_read_ready; // Fetcher read ready signals
    reg [PROGRAM_MEM_DATA_BITS-1:0] fetcher_read_data [NUM_FETCHERS-1:0]; // Fetcher read data
    
    // Device Control Register
    dcr dcr_instance (
        .clk(clk), // Clock
        .reset(reset), // Reset

        .device_control_write_enable(device_control_write_enable), // Write enable
        .device_control_data(device_control_data), // Data input
        .thread_count(thread_count) // Output thread count
    );

    // Data Memory Controller
    controller #(
        .ADDR_BITS(DATA_MEM_ADDR_BITS), // Address width
        .DATA_BITS(DATA_MEM_DATA_BITS), // Data width
        .NUM_CONSUMERS(NUM_LSUS), // Number of consumers (LSUs)
        .NUM_CHANNELS(DATA_MEM_NUM_CHANNELS) // Number of memory channels
    ) data_memory_controller (
        .clk(clk), // Clock
        .reset(reset), // Reset

        .consumer_read_valid(lsu_read_valid), // LSU read valid
        .consumer_read_address(lsu_read_address), // LSU read address
        .consumer_read_ready(lsu_read_ready), // LSU read ready
        .consumer_read_data(lsu_read_data), // LSU read data
        .consumer_write_valid(lsu_write_valid), // LSU write valid
        .consumer_write_address(lsu_write_address), // LSU write address
        .consumer_write_data(lsu_write_data), // LSU write data
        .consumer_write_ready(lsu_write_ready), // LSU write ready

        .mem_read_valid(data_mem_read_valid), // Memory read valid
        .mem_read_address(data_mem_read_address), // Memory read address
        .mem_read_ready(data_mem_read_ready), // Memory read ready
        .mem_read_data(data_mem_read_data), // Memory read data
        .mem_write_valid(data_mem_write_valid), // Memory write valid
        .mem_write_address(data_mem_write_address), // Memory write address
        .mem_write_data(data_mem_write_data), // Memory write data
        .mem_write_ready(data_mem_write_ready) // Memory write ready
    );

    // Program Memory Controller
    controller #(
        .ADDR_BITS(PROGRAM_MEM_ADDR_BITS), // Address width
        .DATA_BITS(PROGRAM_MEM_DATA_BITS), // Data width
        .NUM_CONSUMERS(NUM_FETCHERS), // Number of consumers (fetchers)
        .NUM_CHANNELS(PROGRAM_MEM_NUM_CHANNELS), // Number of memory channels
        .WRITE_ENABLE(0) // Disable write
    ) program_memory_controller (
        .clk(clk), // Clock
        .reset(reset), // Reset

        .consumer_read_valid(fetcher_read_valid), // Fetcher read valid
        .consumer_read_address(fetcher_read_address), // Fetcher read address
        .consumer_read_ready(fetcher_read_ready), // Fetcher read ready
        .consumer_read_data(fetcher_read_data), // Fetcher read data

        .mem_read_valid(program_mem_read_valid), // Memory read valid
        .mem_read_address(program_mem_read_address), // Memory read address
        .mem_read_ready(program_mem_read_ready), // Memory read ready
        .mem_read_data(program_mem_read_data), // Memory read data
    );

    // Dispatcher
    dispatch #(
        .NUM_CORES(NUM_CORES), // Number of cores
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK) // Threads per block
    ) dispatch_instance (
        .clk(clk), // Clock
        .reset(reset), // Reset
        .start(start), // Start signal
        .thread_count(thread_count), // Number of threads to launch
        .core_done(core_done), // Done signals from cores
        .core_start(core_start), // Start signals to cores
        .core_reset(core_reset), // Reset signals to cores
        .core_block_id(core_block_id), // Block IDs for cores
        .core_thread_count(core_thread_count), // Thread counts for cores
        .done(done) // Done signal for kernel
    );

    // Compute Cores
    genvar i; // Generate variable for cores
    generate
        for (i = 0; i < NUM_CORES; i = i + 1) begin : cores // Loop over each core
            // EDA: We create separate signals here to pass to cores because of a requirement
            // by the OpenLane EDA flow (uses Verilog 2005) that prevents slicing the top-level signals
            reg [THREADS_PER_BLOCK-1:0] core_lsu_read_valid; // LSU read valid for this core
            reg [DATA_MEM_ADDR_BITS-1:0] core_lsu_read_address [THREADS_PER_BLOCK-1:0]; // LSU read address for this core
            reg [THREADS_PER_BLOCK-1:0] core_lsu_read_ready; // LSU read ready for this core
            reg [DATA_MEM_DATA_BITS-1:0] core_lsu_read_data [THREADS_PER_BLOCK-1:0]; // LSU read data for this core
            reg [THREADS_PER_BLOCK-1:0] core_lsu_write_valid; // LSU write valid for this core
            reg [DATA_MEM_ADDR_BITS-1:0] core_lsu_write_address [THREADS_PER_BLOCK-1:0]; // LSU write address for this core
            reg [DATA_MEM_DATA_BITS-1:0] core_lsu_write_data [THREADS_PER_BLOCK-1:0]; // LSU write data for this core
            reg [THREADS_PER_BLOCK-1:0] core_lsu_write_ready; // LSU write ready for this core

            // Pass through signals between LSUs and data memory controller
            genvar j; // Generate variable for threads
            for (j = 0; j < THREADS_PER_BLOCK; j = j + 1) begin // Loop over each thread
                localparam lsu_index = i * THREADS_PER_BLOCK + j; // Calculate LSU index
                always @(posedge clk) begin // On clock edge
                    lsu_read_valid[lsu_index] <= core_lsu_read_valid[j]; // Assign read valid
                    lsu_read_address[lsu_index] <= core_lsu_read_address[j]; // Assign read address

                    lsu_write_valid[lsu_index] <= core_lsu_write_valid[j]; // Assign write valid
                    lsu_write_address[lsu_index] <= core_lsu_write_address[j]; // Assign write address
                    lsu_write_data[lsu_index] <= core_lsu_write_data[j]; // Assign write data
                    
                    core_lsu_read_ready[j] <= lsu_read_ready[lsu_index]; // Assign read ready back to core
                    core_lsu_read_data[j] <= lsu_read_data[lsu_index]; // Assign read data back to core
                    core_lsu_write_ready[j] <= lsu_write_ready[lsu_index]; // Assign write ready back to core
                end
            end

            // Compute Core
            core #(
                .DATA_MEM_ADDR_BITS(DATA_MEM_ADDR_BITS), // Data memory address width
                .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS), // Data memory data width
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS), // Program memory address width
                .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS), // Program memory data width
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK), // Threads per block
            ) core_instance (
                .clk(clk), // Clock
                .reset(core_reset[i]), // Reset for this core
                .start(core_start[i]), // Start for this core
                .done(core_done[i]), // Done from this core
                .block_id(core_block_id[i]), // Block ID for this core
                .thread_count(core_thread_count[i]), // Thread count for this core
                
                .program_mem_read_valid(fetcher_read_valid[i]), // Program memory read valid for this core
                .program_mem_read_address(fetcher_read_address[i]), // Program memory read address for this core
                .program_mem_read_ready(fetcher_read_ready[i]), // Program memory read ready for this core
                .program_mem_read_data(fetcher_read_data[i]), // Program memory read data for this core

                .data_mem_read_valid(core_lsu_read_valid), // Data memory read valid for this core
                .data_mem_read_address(core_lsu_read_address), // Data memory read address for this core
                .data_mem_read_ready(core_lsu_read_ready), // Data memory read ready for this core
                .data_mem_read_data(core_lsu_read_data), // Data memory read data for this core
                .data_mem_write_valid(core_lsu_write_valid), // Data memory write valid for this core
                .data_mem_write_address(core_lsu_write_address), // Data memory write address for this core
                .data_mem_write_data(core_lsu_write_data), // Data memory write data for this core
                .data_mem_write_ready(core_lsu_write_ready) // Data memory write ready for this core
            );
        end
    endgenerate
endmodule

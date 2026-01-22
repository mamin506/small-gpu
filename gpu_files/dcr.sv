`default_nettype none                // Disable implicit net declarations for safety
`timescale 1ns/1ns                   // Set simulation time unit and precision

// DEVICE CONTROL REGISTER
// > Used to configure high-level settings
// > In this minimal example, the DCR is used to configure the number of threads to run for the kernel
module dcr (
    input wire clk,                  // Clock input
    input wire reset,                // Synchronous reset input

    input wire device_control_write_enable, // Write enable signal for device control register
    input wire [7:0] device_control_data,  // 8-bit data input for device control
    output wire [7:0] thread_count,        // 8-bit output representing thread count
);
    // Store device control data in dedicated register
    reg [7:0] device_conrol_register;      // 8-bit register to hold device control data
    assign thread_count = device_conrol_register[7:0]; // Assign lower 8 bits to thread_count output

    always @(posedge clk) begin            // Trigger on rising edge of clock
        if (reset) begin                   // If reset is asserted
            device_conrol_register <= 8'b0; // Clear the register to 0
        end else begin
            if (device_control_write_enable) begin // If write enable is asserted
                device_conrol_register <= device_control_data; // Store input data in register
            end
        end
    end
endmodule
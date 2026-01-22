`default_nettype none // Disable implicit net declarations for safety
`timescale 1ns/1ns    // Set simulation time unit and precision

// ARITHMETIC-LOGIC UNIT
// > Executes computations on register values
// > In this minimal implementation, the ALU supports the 4 basic arithmetic operations
// > Each thread in each core has its own ALU
// > ADD, SUB, MUL, DIV instructions are all executed here
module alu (
    input wire clk,                        // Clock signal
    input wire reset,                      // Reset signal (active high)
    input wire enable,                     // Enable signal for ALU operation

    input reg [2:0] core_state,            // Current state of the core

    input reg [1:0] decoded_alu_arithmetic_mux, // Selects arithmetic operation
    input reg decoded_alu_output_mux,           // Selects output mode (arithmetic or NZP flags)

    input reg [7:0] rs,                    // First operand (register source)
    input reg [7:0] rt,                    // Second operand (register target)
    output wire [7:0] alu_out              // ALU output
);
    localparam ADD = 2'b00,                // Local parameter for ADD operation
        SUB = 2'b01,                       // Local parameter for SUB operation
        MUL = 2'b10,                       // Local parameter for MUL operation
        DIV = 2'b11;                       // Local parameter for DIV operation

    reg [7:0] alu_out_reg;                 // Register to hold ALU output
    assign alu_out = alu_out_reg;          // Assign register value to output port

    always @(posedge clk) begin            // Trigger on rising edge of clock
        if (reset) begin                   // If reset is active
            alu_out_reg <= 8'b0;           // Clear ALU output register
        end else if (enable) begin         // If ALU is enabled
            // Calculate alu_out when core_state = EXECUTE
            if (core_state == 3'b101) begin // Check if core is in EXECUTE state
                if (decoded_alu_output_mux == 1) begin // If output mode is NZP flags
                    // Set values to compare with NZP register in alu_out[2:0]
                    alu_out_reg <= {5'b0, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)}; // Set NZP flags in lower 3 bits
                end else begin             // Otherwise, perform arithmetic operation
                    // Execute the specified arithmetic instruction
                    case (decoded_alu_arithmetic_mux) // Select operation based on mux
                        ADD: begin 
                            alu_out_reg <= rs + rt;   // Perform addition
                        end
                        SUB: begin 
                            alu_out_reg <= rs - rt;   // Perform subtraction
                        end
                        MUL: begin 
                            alu_out_reg <= rs * rt;   // Perform multiplication
                        end
                        DIV: begin 
                            alu_out_reg <= rs / rt;   // Perform division
                        end
                    endcase
                end
            end
        end
    end
endmodule

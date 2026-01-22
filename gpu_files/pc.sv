`default_nettype none                // Disable implicit net declarations
`timescale 1ns/1ns                   // Set simulation time unit and precision

// PROGRAM COUNTER
// > Calculates the next PC for each thread to update to (but currently we assume all threads
//   update to the same PC and don't support branch divergence)
// > Currently, each thread in each core has it's own calculation for next PC
// > The NZP register value is set by the CMP instruction (based on >/=/< comparison) to 
//   initiate the BRnzp instruction for branching
module pc #(                         // Start of module definition for 'pc'
    parameter DATA_MEM_DATA_BITS = 8,            // Parameter: width of data memory data bus
    parameter PROGRAM_MEM_ADDR_BITS = 8          // Parameter: width of program memory address bus
) (
    input wire clk,                              // Clock input
    input wire reset,                            // Reset input (active high)
    input wire enable,                           // Enable input for PC update

    // State
    input reg [2:0] core_state,                  // Current state of the core (3 bits)

    // Control Signals
    input reg [2:0] decoded_nzp,                 // Decoded NZP bits for branch condition
    input reg [DATA_MEM_DATA_BITS-1:0] decoded_immediate, // Immediate value from instruction
    input reg decoded_nzp_write_enable,          // Enable signal to write to NZP register
    input reg decoded_pc_mux,                    // Select signal for PC update source

    // ALU Output - used for alu_out[2:0] to compare with NZP register
    input reg [DATA_MEM_DATA_BITS-1:0] alu_out,  // Output from ALU

    // Current & Next PCs
    input reg [PROGRAM_MEM_ADDR_BITS-1:0] current_pc, // Current program counter value
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] next_pc    // Next program counter value
);
    reg [2:0] nzp;                                // NZP register to store condition flags

    always @(posedge clk) begin                   // Always block triggered on rising clock edge
        if (reset) begin                          // If reset is asserted
            nzp <= 3'b0;                          //   Clear NZP register
            next_pc <= 0;                         //   Reset next_pc to 0
        end else if (enable) begin                // If enable is asserted
            // Update PC when core_state = EXECUTE
            if (core_state == 3'b101) begin       //   If core is in EXECUTE state
                if (decoded_pc_mux == 1) begin    //     If PC mux selects branch logic
                    if (((nzp & decoded_nzp) != 3'b0)) begin //       If NZP condition matches
                        // On BRnzp instruction, branch to immediate if NZP case matches previous CMP
                        next_pc <= decoded_immediate;        //         Set next_pc to immediate value
                    end else begin 
                        // Otherwise, just update to PC + 1 (next line)
                        next_pc <= current_pc + 1;           //         Increment PC by 1
                    end
                end else begin 
                    // By default update to PC + 1 (next line)
                    next_pc <= current_pc + 1;               //       Increment PC by 1
                end
            end   

            // Store NZP when core_state = UPDATE   
            if (core_state == 3'b110) begin       //   If core is in UPDATE state
                // Write to NZP register on CMP instruction
                if (decoded_nzp_write_enable) begin           //     If NZP write is enabled
                    nzp[2] <= alu_out[2];                    //       Update NZP[2] with ALU output bit 2
                    nzp[1] <= alu_out[1];                    //       Update NZP[1] with ALU output bit 1
                    nzp[0] <= alu_out[0];                    //       Update NZP[0] with ALU output bit 0
                end
            end      
        end
    end

endmodule                                         // End of module definition

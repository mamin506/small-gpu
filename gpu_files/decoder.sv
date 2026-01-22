`default_nettype none                  // Disable implicit net declarations
`timescale 1ns/1ns                     // Set simulation time unit and precision

// INSTRUCTION DECODER
// > Decodes an instruction into the control signals necessary to execute it
// > Each core has its own decoder
module decoder (
    input wire clk,                    // Clock signal
    input wire reset,                  // Reset signal

    input reg [2:0] core_state,        // Current state of the core (3 bits)
    input reg [15:0] instruction,      // Current instruction to decode (16 bits)
    
    // Instruction Signals
    output reg [3:0] decoded_rd_address,   // Decoded destination register address
    output reg [3:0] decoded_rs_address,   // Decoded source register address
    output reg [3:0] decoded_rt_address,   // Decoded target register address
    output reg [2:0] decoded_nzp,          // Decoded NZP (Negative/Zero/Positive) flags
    output reg [7:0] decoded_immediate,    // Decoded immediate value (8 bits)
    
    // Control Signals
    output reg decoded_reg_write_enable,           // Enable writing to a register
    output reg decoded_mem_read_enable,            // Enable reading from memory
    output reg decoded_mem_write_enable,           // Enable writing to memory
    output reg decoded_nzp_write_enable,           // Enable writing to NZP register
    output reg [1:0] decoded_reg_input_mux,        // Select input to register (2 bits)
    output reg [1:0] decoded_alu_arithmetic_mux,   // Select arithmetic operation (2 bits)
    output reg decoded_alu_output_mux,             // Select operation in ALU
    output reg decoded_pc_mux,                     // Select source of next PC

    // Return (finished executing thread)
    output reg decoded_ret                         // Signal to indicate return from execution
);
    // Define instruction opcodes as local parameters
    localparam NOP = 4'b0000,      // No operation
        BRnzp = 4'b0001,           // Branch if NZP
        CMP = 4'b0010,             // Compare
        ADD = 4'b0011,             // Add
        SUB = 4'b0100,             // Subtract
        MUL = 4'b0101,             // Multiply
        DIV = 4'b0110,             // Divide
        LDR = 4'b0111,             // Load register
        STR = 4'b1000,             // Store register
        CONST = 4'b1001,           // Load constant
        RET = 4'b1111;             // Return

    // Always block triggered on rising edge of clock
    always @(posedge clk) begin 
        if (reset) begin 
            // Reset all decoded outputs to zero
            decoded_rd_address <= 0;
            decoded_rs_address <= 0;
            decoded_rt_address <= 0;
            decoded_immediate <= 0;
            decoded_nzp <= 0;
            decoded_reg_write_enable <= 0;
            decoded_mem_read_enable <= 0;
            decoded_mem_write_enable <= 0;
            decoded_nzp_write_enable <= 0;
            decoded_reg_input_mux <= 0;
            decoded_alu_arithmetic_mux <= 0;
            decoded_alu_output_mux <= 0;
            decoded_pc_mux <= 0;
            decoded_ret <= 0;
        end else begin 
            // Decode only when core_state is DECODE (3'b010)
            if (core_state == 3'b010) begin 
                // Extract register addresses and immediate values from instruction
                decoded_rd_address <= instruction[11:8];    // Bits 11-8: destination register
                decoded_rs_address <= instruction[7:4];     // Bits 7-4: source register
                decoded_rt_address <= instruction[3:0];     // Bits 3-0: target register
                decoded_immediate <= instruction[7:0];      // Bits 7-0: immediate value
                decoded_nzp <= instruction[11:9];           // Bits 11-9: NZP flags

                // Reset all control signals before setting them
                decoded_reg_write_enable <= 0;
                decoded_mem_read_enable <= 0;
                decoded_mem_write_enable <= 0;
                decoded_nzp_write_enable <= 0;
                decoded_reg_input_mux <= 0;
                decoded_alu_arithmetic_mux <= 0;
                decoded_alu_output_mux <= 0;
                decoded_pc_mux <= 0;
                decoded_ret <= 0;

                // Set control signals based on instruction opcode
                case (instruction[15:12])       // Bits 15-12: opcode
                    NOP: begin 
                        // No operation, do nothing
                    end
                    BRnzp: begin 
                        decoded_pc_mux <= 1;    // Set PC mux for branch
                    end
                    CMP: begin 
                        decoded_alu_output_mux <= 1;      // Select compare in ALU
                        decoded_nzp_write_enable <= 1;    // Enable NZP write
                    end
                    ADD: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b00;       // Select ALU result
                        decoded_alu_arithmetic_mux <= 2'b00;  // Select add operation
                    end
                    SUB: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b00;       // Select ALU result
                        decoded_alu_arithmetic_mux <= 2'b01;  // Select subtract operation
                    end
                    MUL: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b00;       // Select ALU result
                        decoded_alu_arithmetic_mux <= 2'b10;  // Select multiply operation
                    end
                    DIV: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b00;       // Select ALU result
                        decoded_alu_arithmetic_mux <= 2'b11;  // Select divide operation
                    end
                    LDR: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b01;       // Select memory input
                        decoded_mem_read_enable <= 1;         // Enable memory read
                    end
                    STR: begin 
                        decoded_mem_write_enable <= 1;        // Enable memory write
                    end
                    CONST: begin 
                        decoded_reg_write_enable <= 1;        // Enable register write
                        decoded_reg_input_mux <= 2'b10;       // Select constant input
                    end
                    RET: begin 
                        decoded_ret <= 1;                     // Signal return
                    end
                endcase
            end
        end
    end
endmodule

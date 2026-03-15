from typing import List, Optional  # Import List and Optional types for type hinting
from .logger import logger         # Import logger from the local logger module

def format_register(register: int) -> str:
    # Format register number to its string representation
    if register < 13:
        return f"R{register}"      # Registers 0-12 are named R0-R12
    if register == 13:
        return f"%blockIdx"        # Register 13 is blockIdx
    if register == 14:
        return f"%blockDim"        # Register 14 is blockDim
    if register == 15:
        return f"%threadIdx"       # Register 15 is threadIdx

def format_instruction(instruction: str) -> str:
    # Format a binary instruction string to its assembly representation
    opcode = instruction[0:4]                          # Extract opcode (first 4 bits)
    rd = format_register(int(instruction[4:8], 2))     # Extract destination register (bits 4-7)
    rs = format_register(int(instruction[8:12], 2))    # Extract source register 1 (bits 8-11)
    rt = format_register(int(instruction[12:16], 2))   # Extract source register 2 (bits 12-15)
    n = "N" if instruction[4] == 1 else ""             # Set N flag if bit 4 is 1
    z = "Z" if instruction[5] == 1 else ""             # Set Z flag if bit 5 is 1
    p = "P" if instruction[6] == 1 else ""             # Set P flag if bit 6 is 1
    imm = f"#{int(instruction[8:16], 2)}"              # Immediate value from bits 8-15

    if opcode == "0000":
        return "NOP"                                   # No operation
    elif opcode == "0001":
        return f"BRnzp {n}{z}{p}, {imm}"               # Branch instruction
    elif opcode == "0010":
        return f"CMP {rs}, {rt}"                       # Compare instruction
    elif opcode == "0011":
        return f"ADD {rd}, {rs}, {rt}"                 # Add instruction
    elif opcode == "0100":
        return f"SUB {rd}, {rs}, {rt}"                 # Subtract instruction
    elif opcode == "0101":
        return f"MUL {rd}, {rs}, {rt}"                 # Multiply instruction
    elif opcode == "0110":
        return f"DIV {rd}, {rs}, {rt}"                 # Divide instruction
    elif opcode == "0111":
        return f"LDR {rd}, {rs}"                       # Load instruction
    elif opcode == "1000":
        return f"STR {rs}, {rt}"                       # Store instruction
    elif opcode == "1001":
        return f"CONST {rd}, {imm}"                    # Constant assignment
    elif opcode == "1111":
        return "RET"                                   # Return instruction
    return "UNKNOWN"                                   # Unknown opcode

def format_core_state(core_state: str) -> str:
    # Map core state binary string to its name
    core_state_map = {
        "000": "IDLE",         # Idle state
        "001": "FETCH",        # Fetch state
        "010": "DECODE",       # Decode state
        "011": "REQUEST",      # Request state
        "100": "WAIT",         # Wait state
        "101": "EXECUTE",      # Execute state
        "110": "UPDATE",       # Update state
        "111": "DONE"          # Done state
    }
    return core_state_map[core_state]   # Return mapped state name

def format_fetcher_state(fetcher_state: str) -> str:
    # Map fetcher state binary string to its name
    fetcher_state_map = {
        "000": "IDLE",         # Idle state
        "001": "FETCHING",     # Fetching state
        "010": "FETCHED"       # Fetched state
    }
    return fetcher_state_map[fetcher_state]   # Return mapped state name

def format_lsu_state(lsu_state: str) -> str:
    # Map LSU state binary string to its name
    lsu_state_map = {
        "00": "IDLE",          # Idle state
        "01": "REQUESTING",    # Requesting state
        "10": "WAITING",       # Waiting state
        "11": "DONE"           # Done state
    }
    return lsu_state_map[lsu_state]           # Return mapped state name

def format_memory_controller_state(controller_state: str) -> str:
    # Map memory controller state binary string to its name
    controller_state_map = {
        "000": "IDLE",             # Idle state
        "010": "READ_WAITING",     # Read waiting state
        "011": "WRITE_WAITING",    # Write waiting state
        "100": "READ_RELAYING",    # Read relaying state
        "101": "WRITE_RELAYING"    # Write relaying state
    }
    return controller_state_map[controller_state]   # Return mapped state name

def format_registers(registers: List[str]) -> str:
    # Format a list of register binary strings to their names and decimal values
    formatted_registers = []                        # List to hold formatted register strings
    for i, reg_value in enumerate(registers):       # Iterate over registers
        decimal_value = int(reg_value, 2)           # Convert binary string to decimal
        reg_idx = 15 - i                            # Register index (reverse order)
        formatted_registers.append(f"{format_register(reg_idx)} = {decimal_value}") # Format string
    formatted_registers.reverse()                   # Reverse to match display order
    return ', '.join(formatted_registers)           # Join all formatted strings

def format_cycle(dut, cycle_id: int, thread_id: Optional[int] = None):
    # Format and log the state of the DUT for a given cycle and optional thread
    logger.debug(f"\n================================== Cycle {cycle_id} ==================================") # Log cycle header

    for core in dut.cores:                          # Iterate over all cores
        # Skip core if thread count is less than required for this core
        if int(str(dut.thread_count.value), 2) <= int(core.i.value) * int(dut.THREADS_PER_BLOCK.value):
            continue

        logger.debug(f"\n+--------------------- Core {core.i.value} ---------------------+") # Log core header

        instruction = str(core.core_instance.instruction.value) # Get instruction for this core
        for thread in core.core_instance.threads:               # Iterate over threads in core
            # Check if thread is enabled
            if int(thread.i.value) < int(str(core.core_instance.thread_count.value), 2):
                block_idx = int(core.core_instance.block_id.value)           # Get block index
                block_dim = int(core.core_instance.THREADS_PER_BLOCK.value)   # Get block dimension
                thread_idx = int(thread.register_instance.THREAD_ID.value)   # Get thread index
                idx = block_idx * block_dim + thread_idx                # Calculate global thread index

                rs = int(str(thread.register_instance.rs.value), 2)     # Get RS register value
                rt = int(str(thread.register_instance.rt.value), 2)     # Get RT register value

                reg_input_mux = int(str(core.core_instance.decoded_reg_input_mux.value), 2) # Get register input mux
                alu_out = int(str(thread.alu_instance.alu_out.value), 2)                   # Get ALU output value
                lsu_out = int(str(thread.lsu_instance.lsu_out.value), 2)                   # Get LSU output value
                constant = int(str(core.core_instance.decoded_immediate.value), 2)         # Get constant value

                # If thread_id is not specified or matches current thread
                if (thread_id is None or thread_id == idx):
                    logger.debug(f"\n+-------- Thread {idx} --------+")                     # Log thread header

                    logger.debug("PC:", int(str(core.core_instance.current_pc.value), 2))  # Log program counter
                    logger.debug("Instruction:", format_instruction(instruction))           # Log formatted instruction
                    logger.debug("Core State:", format_core_state(str(core.core_instance.core_state.value))) # Log core state
                    logger.debug("Fetcher State:", format_fetcher_state(str(core.core_instance.fetcher_state.value))) # Log fetcher state
                    logger.debug("LSU State:", format_lsu_state(str(thread.lsu_instance.lsu_state.value))) # Log LSU state
                    logger.debug("Registers:", format_registers([str(item.value) for item in thread.register_instance.registers])) # Log registers
                    logger.debug(f"RS = {rs}, RT = {rt}")                                  # Log RS and RT values

                    if reg_input_mux == 0:
                        logger.debug("ALU Out:", alu_out)                                  # Log ALU output if mux is 0
                    if reg_input_mux == 1:
                        logger.debug("LSU Out:", lsu_out)                                  # Log LSU output if mux is 1
                    if reg_input_mux == 2:
                        logger.debug("Constant:", constant)                                # Log constant if mux is 2

        logger.debug("Core Done:", str(core.core_instance.done.value))                     # Log core done status
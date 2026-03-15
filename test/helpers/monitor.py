#! python

def log_core_state(dut, logger, cycle):
    """
    Peeks inside Core 0 to see what the hardware is doing.
    This is equivalent to attaching an Oscilloscope to the chip.
    """
    # Access internal signals via hierarchy: GPU -> Core[0] -> Internal Wires
    # dut.genblk1[0] accesses the first generated Core instance
    core = dut.genblk1[0].core_instance
    
    # 1. Read The Program Counter (Where are we?)
    pc = int(core.current_pc.value)
    
    # 2. Read the State Machine (What are we doing?)
    # Map binary state to human readable string
    states = {0:"IDLE", 1:"FETCH", 2:"DECODE", 3:"REQUEST", 4:"WAIT", 5:"EXECUTE", 6:"UPDATE", 7:"DONE"}
    state_val = int(core.core_state.value)
    state_str = states.get(state_val, "UNKNOWN")
    
    # 3. Spy on Thread 0's Registers (The "Backpack")
    # We look at Register 0 (Accumulator) and Register 15 (Thread ID)
    # Note: Using hierarchy to reach inside the generated loop for Thread 0
    thread0_regs = core.genblk2[0].register_instance.registers
    r0_val = int(thread0_regs[0].value) # R0 (General Purpose)
    r15_val = int(thread0_regs[15].value) # ThreadID (System)
    
    # 4. Spy on the Instruction
    instr = int(core.instruction.value)
    
    # Log the snapshot
    logger.info(f"[Cycle {cycle:03d}] PC: {pc:02d} | State: {state_str:<8} | Instr: {hex(instr)} | T0_R0: {r0_val}")


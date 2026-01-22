import cocotb  # Import the cocotb library for coroutine-based testbenches
from cocotb.triggers import RisingEdge  # Import RisingEdge trigger for clock synchronization
from .helpers.setup import setup  # Import setup helper function for initializing test
from .helpers.memory import Memory  # Import Memory helper class for memory operations
from .helpers.format import format_cycle  # Import format_cycle helper for cycle formatting
from .helpers.logger import logger  # Import logger helper for logging information

@cocotb.test()  # Decorator to mark this function as a cocotb test
async def test_matadd(dut):  # Define asynchronous test function with device under test (dut)
    # Program Memory
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")  # Create program memory instance with 8 address bits, 16 data bits, 1 channel
    program = [  # Define program instructions for matrix addition kernel
        0b0101000011011110, # MUL R0, %blockIdx, %blockDim
        0b0011000000001111, # ADD R0, R0, %threadIdx         ; i = blockIdx * blockDim + threadIdx
        0b1001000100000000, # CONST R1, #0                   ; baseA (matrix A base address)
        0b1001001000001000, # CONST R2, #8                   ; baseB (matrix B base address)
        0b1001001100010000, # CONST R3, #16                  ; baseC (matrix C base address)
        0b0011010000010000, # ADD R4, R1, R0                 ; addr(A[i]) = baseA + i
        0b0111010001000000, # LDR R4, R4                     ; load A[i] from global memory
        0b0011010100100000, # ADD R5, R2, R0                 ; addr(B[i]) = baseB + i
        0b0111010101010000, # LDR R5, R5                     ; load B[i] from global memory
        0b0011011001000101, # ADD R6, R4, R5                 ; C[i] = A[i] + B[i]
        0b0011011100110000, # ADD R7, R3, R0                 ; addr(C[i]) = baseC + i
        0b1000000001110110, # STR R7, R6                     ; store C[i] in global memory
        0b1111000000000000, # RET                            ; end of kernel
    ]

    # Data Memory
    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")  # Create data memory instance with 8 address bits, 8 data bits, 4 channels
    data = [  # Define initial data for matrices A and B
        0, 1, 2, 3, 4, 5, 6, 7, # Matrix A (1 x 8)
        0, 1, 2, 3, 4, 5, 6, 7  # Matrix B (1 x 8)
    ]

    # Device Control
    threads = 8  # Set number of threads for parallel execution

    await setup(  # Await setup helper to initialize memories and device state
        dut=dut,
        program_memory=program_memory,
        program=program,
        data_memory=data_memory,
        data=data,
        threads=threads
    )

    data_memory.display(24)  # Display first 24 entries of data memory for debugging

    cycles = 0  # Initialize cycle counter
    while dut.done.value != 1:  # Loop until device signals completion
        data_memory.run()  # Run data memory operations for current cycle
        program_memory.run()  # Run program memory operations for current cycle

        await cocotb.triggers.ReadOnly()  # Await ReadOnly trigger to sample signals
        format_cycle(dut, cycles)  # Format and display current cycle information
        
        await RisingEdge(dut.clk)  # Wait for rising edge of clock
        cycles += 1  # Increment cycle counter

    logger.info(f"Completed in {cycles} cycles")  # Log completion and cycle count
    data_memory.display(24)  # Display data memory after execution for result inspection

    expected_results = [a + b for a, b in zip(data[0:8], data[8:16])]  # Compute expected results for matrix addition
    for i, expected in enumerate(expected_results):  # Iterate over expected results
        result = data_memory.memory[i + 16]  # Read result from data memory (C matrix starts at address 16)
        assert result == expected, f"Result mismatch at index {i}: expected {expected}, got {result}"  # Assert correctness of result
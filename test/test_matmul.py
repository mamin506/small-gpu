import cocotb  # Import cocotb for coroutine-based testbench
from cocotb.triggers import RisingEdge  # Import RisingEdge trigger for clock synchronization
from .helpers.setup import setup  # Import setup helper for initializing memories and device
from .helpers.memory import Memory  # Import Memory helper class for memory abstraction
from .helpers.format import format_cycle  # Import format_cycle for cycle formatting/logging
from .helpers.logger import logger  # Import logger for logging test information

@cocotb.test()  # Mark this function as a cocotb test
async def test_matadd(dut):  # Define asynchronous test function with device under test (dut)
    # Program Memory
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")  # Create program memory with 8-bit address, 16-bit data, 1 channel
    program = [  # Define program instructions (machine code for matrix multiplication kernel)
        0b0101000011011110, # MUL R0, %blockIdx, %blockDim
        0b0011000000001111, # ADD R0, R0, %threadIdx         ; i = blockIdx * blockDim + threadIdx
        0b1001000100000001, # CONST R1, #1                   ; increment
        0b1001001000000010, # CONST R2, #2                   ; N (matrix inner dimension)
        0b1001001100000000, # CONST R3, #0                   ; baseA (matrix A base address)
        0b1001010000000100, # CONST R4, #4                   ; baseB (matrix B base address)
        0b1001010100001000, # CONST R5, #8                   ; baseC (matrix C base address)
        0b0110011000000010, # DIV R6, R0, R2                 ; row = i // N
        0b0101011101100010, # MUL R7, R6, R2
        0b0100011100000111, # SUB R7, R0, R7                 ; col = i % N
        0b1001100000000000, # CONST R8, #0                   ; acc = 0
        0b1001100100000000, # CONST R9, #0                   ; k = 0
                            # LOOP:
        0b0101101001100010, #   MUL R10, R6, R2
        0b0011101010101001, #   ADD R10, R10, R9
        0b0011101010100011, #   ADD R10, R10, R3             ; addr(A[i]) = row * N + k + baseA
        0b0111101010100000, #   LDR R10, R10                 ; load A[i] from global memory
        0b0101101110010010, #   MUL R11, R9, R2
        0b0011101110110111, #   ADD R11, R11, R7
        0b0011101110110100, #   ADD R11, R11, R4             ; addr(B[i]) = k * N + col + baseB
        0b0111101110110000, #   LDR R11, R11                 ; load B[i] from global memory
        0b0101110010101011, #   MUL R12, R10, R11
        0b0011100010001100, #   ADD R8, R8, R12              ; acc = acc + A[i] * B[i]
        0b0011100110010001, #   ADD R9, R9, R1               ; increment k
        0b0010000010010010, #   CMP R9, R2
        0b0001100000001100, #   BRn LOOP                     ; loop while k < N
        0b0011100101010000, # ADD R9, R5, R0                 ; addr(C[i]) = baseC + i 
        0b1000000010011000, # STR R9, R8                     ; store C[i] in global memory
        0b1111000000000000  # RET                            ; end of kernel
    ]

    # Data Memory
    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")  # Create data memory with 8-bit address, 8-bit data, 4 channels
    data = [  # Define initial data for memory (matrices A and B)
        1, 2, 3, 4, # Matrix A (2 x 2)
        1, 2, 3, 4, # Matrix B (2 x 2)
    ]

    # Device Control
    threads = 4  # Set number of threads for parallel execution

    await setup(  # Initialize device, memories, and load program/data
        dut=dut,
        program_memory=program_memory,
        program=program,
        data_memory=data_memory,
        data=data,
        threads=threads
    )

    data_memory.display(12)  # Display first 12 entries of data memory for debugging

    cycles = 0  # Initialize cycle counter
    while dut.done.value != 1:  # Run simulation until device signals completion
        data_memory.run()  # Simulate data memory behavior for this cycle
        program_memory.run()  # Simulate program memory behavior for this cycle

        await cocotb.triggers.ReadOnly()  # Wait for read-only phase of simulation
        format_cycle(dut, cycles, thread_id=1)  # Format and log current cycle

        await RisingEdge(dut.clk)  # Wait for rising edge of clock
        cycles += 1  # Increment cycle counter

    logger.info(f"Completed in {cycles} cycles")  # Log total cycles taken
    data_memory.display(12)  # Display first 12 entries of data memory after execution

    # Assuming the matrices are 2x2 and the result is stored starting at address 9
    matrix_a = [data[0:2], data[2:4]]  # Extract matrix A as 2x2 from data
    matrix_b = [data[4:6], data[6:8]]  # Extract matrix B as 2x2 from data
    expected_results = [  # Compute expected results for matrix multiplication
        matrix_a[0][0] * matrix_b[0][0] + matrix_a[0][1] * matrix_b[1][0],  # C[0,0]
        matrix_a[0][0] * matrix_b[0][1] + matrix_a[0][1] * matrix_b[1][1],  # C[0,1]
        matrix_a[1][0] * matrix_b[0][0] + matrix_a[1][1] * matrix_b[1][0],  # C[1,0]
        matrix_a[1][0] * matrix_b[0][1] + matrix_a[1][1] * matrix_b[1][1],  # C[1,1]
    ]
    for i, expected in enumerate(expected_results):  # Iterate over expected results
        result = data_memory.memory[i + 8]  # Read result from data memory (results start at address 9)
        assert result == expected, f"Result mismatch at index {i}: expected {expected}, got {result}"  # Assert result matches expected value


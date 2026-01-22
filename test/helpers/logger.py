import datetime  # Import the datetime module to work with dates and times

class Logger:  # Define a Logger class for logging messages to a file
    def __init__(self, level="debug"):  # Initialize the Logger instance with a log level
        # Create a log filename with the current date and time for uniqueness
        self.filename = f"test/logs/log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        self.level = level  # Set the logging level (default is "debug")

    def debug(self, *messages):  # Method to log debug messages
        if self.level == "debug":  # Only log if the level is set to "debug"
            self.info(*messages)  # Delegate to the info method to write the message

    def info(self, *messages):  # Method to log info messages
        # Combine all message arguments into a single string separated by spaces
        full_message = ' '.join(str(message) for message in messages)
        # Open the log file in append mode
        with open(self.filename, "a") as log_file:
            log_file.write(full_message + "\n")  # Write the message followed by a newline

# Create a Logger instance with the log level set to "debug"
logger = Logger(level="debug")
#!/usr/bin/env python3
import subprocess
import sys


def generate_requirements_pip():
    """
    Generate requirements.txt using pip freeze method.
    Most straightforward and commonly used approach.

    Returns:
        str: Path to generated requirements file
    """
    try:
        output_file = '.requirements'

        # Use subprocess to run pip freeze
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True,
            text=True
        )

        # Write requirements to file
        with open(output_file, 'w') as f:
            f.write(result.stdout)

        print(f"Requirements generated successfully: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error generating requirements: {e}")
        return None


def main():
    generate_requirements_pip()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import sys
import subprocess
import os


class PythonRequirementsGenerator:
    @staticmethod
    def generate_requirements_pip():
        """
        Generate requirements.txt using pip freeze method.
        Most straightforward and commonly used approach.

        Returns:
            str: Path to generated requirements file
        """
        try:
            output_file = 'log/requirements.txt'

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

    @staticmethod
    def generate_requirements_pipreqs():
        """
        Generate requirements.txt using pipreqs.
        Analyzes project imports for more precise dependency tracking.

        Returns:
            str: Path to generated requirements file
        """
        try:
            # Ensure pipreqs is installed
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pipreqs'], check=True)

            output_file = 'log/requirements.txt'

            # Use pipreqs to generate requirements based on imports
            result = subprocess.run(
                [sys.executable, '-m', 'pipreqs', '.', '--force', '--savepath', output_file],
                capture_output=True,
                text=True
            )

            print(f"Requirements generated successfully: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            print(f"Error running pipreqs: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error generating requirements: {e}")
            return None

    @staticmethod
    def generate_requirements_poetry():
        """
        Generate requirements.txt using Poetry.
        Ideal for projects managed with Poetry dependency management.

        Returns:
            str: Path to generated requirements file
        """
        try:
            output_file = 'log/requirements.txt'

            # Export Poetry dependencies to requirements.txt
            result = subprocess.run(
                ['poetry', 'export', '-f', 'requirements.txt', '--output', output_file],
                capture_output=True,
                text=True
            )

            print(f"Requirements generated successfully: {output_file}")
            return output_file

        except FileNotFoundError:
            print("Poetry is not installed. Please install Poetry first.")
            return None
        except Exception as e:
            print(f"Error generating requirements with Poetry: {e}")
            return None


def main():
    print("Python Requirements Generation Methods:")
    print("1. pip freeze (Standard)")
    print("2. pipreqs (Import-based)")
    print("3. Poetry Export")

    choice = input("Select method (1/2/3): ").strip()

    if choice == '1':
        PythonRequirementsGenerator.generate_requirements_pip()
    elif choice == '2':
        PythonRequirementsGenerator.generate_requirements_pipreqs()
    elif choice == '3':
        PythonRequirementsGenerator.generate_requirements_poetry()
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == '__main__':
    main()

# Post-Generation Best Practices:
# 1. Review the generated requirements.txt manually
# 2. Consider specifying version constraints (==, >=, etc.)
# 3. Remove unnecessary or system-specific packages
# 4. Use virtual environments to isolate project dependencies
# 5. Commit requirements.txt to version control
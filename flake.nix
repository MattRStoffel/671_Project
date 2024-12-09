{
  description = "A flake for a Python environment with PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311; # Or your desired Python version
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python  # Include the base Python package
            (python.withPackages (ps: with ps; [
              torch
              pandas
              ollama
              nltk
              transformers
              matplotlib
              py-cpuinfo
            ]))  # Add the Python environment with packages
          ];
          # Automatically run and manage `ollama serve`
          shellHook = ''
            echo "Starting Ollama service..."
            ollama serve &  # Start Ollama in the background
            BG_PID=$!        # Capture the PID of the process

            # Ensure the service stops when exiting the shell
            trap 'echo "Stopping Ollama service..."; kill $BG_PID' EXIT
          '';
        };
      }
    );
}


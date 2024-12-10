{
  description = "A flake for a Python environment with PyTorch and Ollama";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311; # Or your desired Python version
	ollama_s = pkgs.ollama;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
	    ollama_s
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
        };
      }
    );
}


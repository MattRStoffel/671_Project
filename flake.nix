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


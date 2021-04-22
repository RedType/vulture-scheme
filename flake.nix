# special thanks to github:srid/rust-nix-template
# for providing the backbone of this file

{
  description = "A scheme for scheming vultures";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crate2nix = {
      url = "github:kolloch/crate2nix";
      flake = false;
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, utils, rust-overlay, crate2nix, ... }:
    let
      name = "vulture-scheme";
    in utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlay
            (self: super: {
              rustc = self.rust-bin.stable.latest.default;
              cargo = self.rust-bin.stable.latest.default;
            })
          ];
        };
        inherit (import "${crate2nix}/tools.nix" { inherit pkgs; }) generatedCargoNix;

        project = pkgs.callPackage (generatedCargoNix {
          inherit name;
          src = ./.;
        })
        {
          # crate overrides go here
          defaultCrateOverrides = pkgs.defaultCrateOverrides // {
            ${name} = oldAttrs: {
              inherit buildInputs nativeBuildInputs;
            } // buildEnvVars;
          };
        };

      buildInputs = with pkgs; [ openssl.dev ];
      nativeBuildInputs = with pkgs; [ rustc cargo pkgconfig nixpkgs-fmt lldb ];
      buildEnvVars = {
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
      };
    in rec {
      packages.${name} = project.rootCrate.build;

      # `nix build`
      defaultPackage = packages.${name};

      # `nix run`
      apps.${name} = utils.lib.mkApp {
        inherit name;
        drv = packages.${name};
      };
      defaultApp = apps.${name};

      # `nix develop`
      devShell = pkgs.mkShell {
        inherit buildInputs nativeBuildInputs;
        RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
      } // buildEnvVars;
    });
}

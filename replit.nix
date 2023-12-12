{pkgs}: {
  deps = [
    pkgs.libxcrypt
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.libxcrypt
    ];
  };
}

import installer

dependencies = ['pygltflib', 'trimesh']
for dependency in dependencies:
    if not installer.installed(dependency, reload=False, quiet=True):
        installer.install(dependency, ignore=False)

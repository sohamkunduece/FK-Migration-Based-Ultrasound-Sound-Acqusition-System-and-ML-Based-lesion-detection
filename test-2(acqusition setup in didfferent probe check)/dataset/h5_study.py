import h5py

file_path = "YOUR FILE PATH HERE"
def inspect_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group]   /{name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[Dataset] /{name} | Shape: {obj.shape} | Type: {obj.dtype}")
    
    if obj.attrs:
        print(f"    ------attributes: {dict(obj.attrs)}")

# Unindent this block so it runs outside the function
with h5py.File(file_path, 'r') as f:
    if f.attrs:
        print(f"[Root attributes]: {dict(f.attrs)}")
    f.visititems(inspect_structure)

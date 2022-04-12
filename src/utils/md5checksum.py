import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def write_checksum(file_path, file_path_checksum):
    md5_checksum = md5(file_path)
    with open(file_path_checksum, "w") as file:
        file.write(md5_checksum)

def read_checksum(file_path_checksum):
    with open(file_path_checksum, 'r') as file:
        data = file.read().replace('\n', '')
    return data.strip()

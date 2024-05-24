import os
import paramiko
from scp import SCPClient
import time


def create_remote_dir(ssh, directory):
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {directory}")
    stdout.channel.recv_exit_status()

def send_files_scp(config, local_dir, remote_dir, file_extension):
    host = config['host']
    port = config['port']
    username = config['username']
    password = config['password']

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname=host, port=port, username=username, password=password)

    create_remote_dir(ssh, remote_dir)

    for filename in os.listdir(local_dir):
        if filename.endswith(file_extension):
            local_file_path = os.path.join(local_dir, filename)

            with SCPClient(ssh.get_transport()) as scp:
                scp.put(local_file_path, os.path.join(remote_dir, filename))

    ssh.close()

def check_and_download_files(config, remote_dir, local_dir, endstr):
    host = config['host']
    port = config['port']
    username = config['username']
    password = config['password']
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    finish_trans = False

    try:
        ssh.connect(hostname=host, port=port, username=username, password=password)

        stdin, stdout, stderr = ssh.exec_command(f"ls {remote_dir}")
        files = stdout.readlines()

        pt_files = [file.strip() for file in files if file.strip().endswith(endstr)]

        if pt_files:
            print("find target files!")
            for i in range(60):
                print(f"Waiting... {i+1} seconds passed")
                time.sleep(1)
            
            stdin, stdout, stderr = ssh.exec_command(f"ls {remote_dir}")
            files = stdout.readlines()
            pt_files = [file.strip() for file in files if file.strip().endswith(endstr)]

            with SCPClient(ssh.get_transport()) as scp:
                for file in pt_files:
                    remote_file_path = f"{remote_dir}/{file}"
                    local_file_path = f"{local_dir}/{file}"
                    scp.get(remote_file_path, local_file_path)
                    print(f"Downloaded {file} to {local_dir}")
            finish_trans = True
        else:
            print("No {} files found in the remote directory.".format(endstr))
            finish_trans = False
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

    return finish_trans
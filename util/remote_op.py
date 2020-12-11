import paramiko, os


def remote_scp(action, host_ip, remote_path, local_path, username, password):
    ssh_port = 22
    # try:
    conn = paramiko.Transport((host_ip, ssh_port))
    conn.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(conn)
    if action == 'remoteRead':
        if not local_path:
            fileName = os.path.split(remote_path)
            local_path = os.path.join('/tmp', fileName[1])
        sftp.get(remote_path, local_path)

    if action == "remoteWrite":
        sftp.put(local_path, remote_path)

    conn.close()
    return True

    # except Exception:
    #     print('error')
    #     return False


if __name__ == '__main__':
    from multiprocessing import Process
    import time

    path = ''
    remote_path = ''


    def my_copy():
        remote_scp(action='remoteWrite',
                   host_ip="",
                   remote_path=remote_path,
                   local_path=path,
                   username='',
                   password='')


    copy_p = Process(target=my_copy)
    copy_p.start()
    time.sleep(30)

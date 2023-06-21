from tqdm.notebook import tqdm as tqdm
from matplotlib import pyplot as plt
import torch
import datetime
import os
import numpy as np

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


# 네트워크 불러오기
def load_latest(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [x for x in ckpt_lst if x.endswith('.pth')]
    
    if len(ckpt_lst) == 0:
        epoch = 0
        return net, optim, epoch

    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def load_manually(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        raise ValueError(f"Model file '{ckpt_dir}' does not exist.")

    dict_model = torch.load(ckpt_dir)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(os.path.basename(ckpt_dir).split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def set_dir(params_json, base_dir):
    # Create the base directory if it doesn't exist already
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create a text file that introduces the parameter settings in the base directory
    with open(os.path.join(base_dir, 'parameter_settings.txt'), 'w') as f:
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        f.write(f"Current Time: {current_time_str}\n")
        f.write(f"Activation String: {params_json['activation_string']}\n")
        f.write(f"Omega String: {params_json['omega_string']}\n")
        f.write(f"Residual String: {params_json['residual_string']}\n")
        f.write(f"Optimizer Name: {params_json['optimizer_function']}\n")
        f.write(f"Loss: {params_json['loss']}\n")
        f.write(f"Learning Rate: {params_json['learning_rate']}\n")
        f.write(f"Num Epoch: {params_json['num_epoch']}\n")
        f.write(f"Milestone: {params_json['milestone']}\n")
        f.write(f"Gamma: {params_json['gamma_optim']}\n")
        f.write(f"Batch Size for Training: {params_json['batchsize_train']}\n")
        f.write(f"Batch Size for Validation: {params_json['batchsize_val']}\n")
        f.write(f"Checkpoint Directory to Load Manually: {params_json['ckpt_dir_to_load_manually']}\n")
        f.write(f"Checkpoint Directory to Save: {params_json['ckpt_dir_save']}\n")
        f.write(f"Log Directory: {params_json['log_dir']}\n")
        f.write(f"Patch Size: {params_json['patch_size']}\n")
        f.write(f"Data Directory for Training: {params_json['datadir_train']}\n")
        f.write(f"Data Directory for Validation: {params_json['datadir_val']}\n")
        f.write(f"Device: {params_json['device']}\n")
        f.write(f"Step to Save: {params_json['step_to_save']}\n")
        f.write(f"QF Candidates: {params_json['QF_candidates']}\n")




def parameter_printer(params_json):
    print('#############################################')
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Current Time: {current_time_str}")
    print(f"Activation String: {params_json['activation_string']}")
    print(f"Omega String: {params_json['omega_string']}")
    print(f"Residual String: {params_json['residual_string']}")
    print(f"Optimizer Name: {params_json['optimizer_function']}")
    print(f"Loss: {params_json['loss']}")
    print(f"Learning Rate: {params_json['learning_rate']}")
    print(f"Num Epoch: {params_json['num_epoch']}")
    print(f"Milestone: {params_json['milestone']}")
    print(f"Gamma: {params_json['gamma_optim']}")
    print(f"Batch Size for Training: {params_json['batchsize_train']}")
    print(f"Batch Size for Validation: {params_json['batchsize_val']}")
    print(f"Base Directory: {params_json['base_dir']}")
    print(f"Checkpoint Directory to Load Manually: {params_json['ckpt_dir_to_load_manually']}")
    print(f"Checkpoint Directory to Save: {params_json['ckpt_dir_save']}")
    print(f"Log Directory: {params_json['log_dir']}")
    print(f"Patch Size: {params_json['patch_size']}")
    print(f"Data Directory for Training: {params_json['datadir_train']}")
    print(f"Data Directory for Validation: {params_json['datadir_val']}")
    print(f"Device: {params_json['device']}")
    print(f"Step to Save: {params_json['step_to_save']}")
    print(f"QF Candidates: {params_json['QF_candidates']}")
    
    print('#############################################\n\n\n\n')

    
def write_successed_msg(params_json):
    msg_body = "The code has finished successfully for the following configuration:\n\n"
    msg_body += f"Activation Function: {params_json['activation_string']}\n"
    msg_body += f"Omega String: {params_json['omega_string']}\n"
    msg_body += f"Residual String: {params_json['residual_string']}\n"
    msg_body += f"Optimizer Name: {params_json['optimizer_function']}\n"
    msg_body += f"Loss: {params_json['loss']}\n"
    msg_body += f"Learning Rate: {params_json['learning_rate']}\n"
    msg_body += f"Num Epoch: {params_json['num_epoch']}\n"
    msg_body += f"Milestone: {params_json['milestone']}\n"
    msg_body += f"Gamma: {params_json['gamma_optim']}\n"
    msg_body += f"Batch Size for Training: {params_json['batchsize_train']}\n"
    msg_body += f"Batch Size for Validation: {params_json['batchsize_val']}\n"
    msg_body += f"Base Directory: {params_json['base_dir']}\n"
    msg_body += f"Checkpoint Directory to Load Manually: {params_json['ckpt_dir_to_load_manually']}\n"
    msg_body += f"Checkpoint Directory to Save: {params_json['ckpt_dir_save']}\n"
    msg_body += f"Log Directory: {params_json['log_dir']}\n"
    msg_body += f"Patch Size: {params_json['patch_size']}\n"
    msg_body += f"Data Directory for Training: {params_json['datadir_train']}\n"
    msg_body += f"Data Directory for Validation: {params_json['datadir_val']}\n"
    msg_body += f"Device: {params_json['device']}\n"
    msg_body += f"Step to Save: {params_json['step_to_save']}\n"
    msg_body += f"QF Candidates: {params_json['QF_candidates']}\n"
    
    return msg_body


def write_error_msg(params_json, error):
    msg_body = "The following error message was received:\n"
    msg_body += f"{error}\n\n"
    msg_body += "Details of the parameter setting at the time of error:\n"
    msg_body += f"- Activation Function: {params_json['activation_string']}\n"
    msg_body += f"- Omega String: {params_json['omega_string']}\n"
    msg_body += f"- Residual String: {params_json['residual_string']}\n"
    msg_body += f"- Optimizer Name: {params_json['optimizer_function']}\n"
    msg_body += f"- Loss: {params_json['loss']}\n"
    msg_body += f"- Learning Rate: {params_json['learning_rate']}\n"
    msg_body += f"- Num Epoch: {params_json['num_epoch']}\n"
    msg_body += f"- Milestone: {params_json['milestone']}\n"
    msg_body += f"- Gamma: {params_json['gamma_optim']}\n"
    msg_body += f"- Batch Size for Training: {params_json['batchsize_train']}\n"
    msg_body += f"- Batch Size for Validation: {params_json['batchsize_val']}\n"
    msg_body += f"- Base Directory: {params_json['base_dir']}\n"
    msg_body += f"- Checkpoint Directory to Load Manually: {params_json['ckpt_dir_to_load_manually']}\n"
    msg_body += f"- Checkpoint Directory to Save: {params_json['ckpt_dir_save']}\n"
    msg_body += f"- Log Directory: {params_json['log_dir']}\n"
    msg_body += f"- Patch Size: {params_json['patch_size']}\n"
    msg_body += f"- Data Directory for Training: {params_json['datadir_train']}\n"
    msg_body += f"- Data Directory for Validation: {params_json['datadir_val']}\n"
    msg_body += f"- Device: {params_json['device']}\n"
    msg_body += f"- Step to Save: {params_json['step_to_save']}\n"
    msg_body += f"- QF Candidates: {params_json['QF_candidates']}\n"
    
    return msg_body



def send_email(subject, body, sender_email='jihyoun0403@gmail.com', recipient_email='jihyoun0403@gmail.com', password='tdautijomcfageko', smtp_server="smtp.gmail.com", port=587):
    """
    A function that sends an email using the SMTP protocol.
    
    Args:
    - subject: A string representing the subject of the email.
    - body: A string representing the body of the email.
    - sender_email: A string representing the email address of the sender. Default is 'jihyoun0403@gmail.com'.
    - recipient_email: A string representing the email address of the recipient. Default is 'jihyoun0403@gmail.com'.
    - password: A string representing the password of the sender's email address. Default is 'tdautijomcfageko'.
    - smtp_server: A string representing the SMTP server to use. Default is 'smtp.gmail.com'.
    - port: An integer representing the port number for the SMTP server. Default is 587.
    
    Returns:
    - None
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Create a MIME message object
    message = MIMEMultipart()
    
    # Set the "From" field to the sender's email address
    message["From"] = sender_email
    
    # Set the "To" field to the recipient's email address
    message["To"] = recipient_email
    
    # Set the "Subject" field to the subject of the email
    message["Subject"] = subject

    # Attach the body of the email as a plain text MIME part
    message.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server and start the TLS encryption
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        
        # Log in to the SMTP server using the sender's email address and password
        server.login(sender_email, password)
        
        # Convert the message to a string and send it to the recipient
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)

        

def check_model_params_changed(model, saved_params):
    """
    Checks if the parameter values of a PyTorch model have changed since they were last saved.
    
    Args:
        model (torch.nn.Module): The PyTorch model to check.
        saved_params (dict): A dictionary of parameter values previously saved using the `state_dict` method.
        
    Returns:
        bool: True if the parameter values have changed, False otherwise.
    """
    current_params = model.state_dict()
    for param_name, param_value in current_params.items():
        if param_name in saved_params and not torch.equal(param_value, saved_params[param_name]):
            return True
    return False



if __name__ == '__main__':
    send_email('test subject', 'test body')
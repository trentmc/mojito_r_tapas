#!/usr/bin/env python

#  Copyright (c) 2005, Corey Goldberg
#
#  SSHController.py is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.

 
"""

@author: Corey Goldberg
@copyright: (C) 2005 Corey Goldberg
@license: GNU General Public License (GPL)
"""


import pyssh


class SSHController:
        
    """Connect to remote host with SSH and issue commands.
    
    This is a facade/wrapper that uses PySSH to spawn and control an SSH client. 
    You must have OpenSSH installed. 
        
    @ivar host_name: Host name or IP address
    @ivar user_name: User name 
    @ivar password: Password
    @ivar prompt: Command prompt (or partial string matching the end of the prompt)
    @ivar ssh: Instance of a pyssh.Ssh object
    """
        
    def __init__(self, host_name, user_name, password, prompt):
        """
            
        @param host_name: Host name or IP address
        @param user_name: User name 
        @param password: Password
        @param prompt: Command prompt (or partial string matching the end of the prompt)
        """
            
        self.host_name = host_name
        self.user_name = user_name
        self.password = password
        self.prompt = prompt
        self.port = '22'  #default SSH port
        self.ssh = None
        
        
    def login(self):
        """Connect to a remote host and login.
            
        """
        self.ssh = pyssh.Ssh(self.user_name, self.host_name, self.port)
        self.ssh.login(self.password)

        
    def run_command(self, command):
        """Run a command on the remote host.
            
        @param command: Unix command
        @return: Command output
        @rtype: String
        """ 
        response = self.ssh.sendcmd(command)
        return self.__strip_output(command, response)
        
    
    def logout(self):
        """Close the connection to the remote host.
            
        """
        self.ssh.logout()
        
        
    def run_atomic_command(self, command):
        """Connect to a remote host, login, run a command, and close the connection.
            
        @param command: Unix command
        @return: Command output
        @rtype: String
        """
        self.login()
        command_output = self.run_command(command)
        self.logout()
        return command_output

        
    def __strip_output(self, command, response):
        """Strip everything from the response except the actual command output.
            
        @param command: Unix command        
        @param response: Command output
        @return: Stripped output
        @rtype: String
        """
        lines = response.splitlines()
        # if our command was echoed back, remove it from the output
        if command in lines[0]:
            lines.pop(0)
        # remove the last element, which is the prompt being displayed again
        lines.pop()
        # append a newline to each line of output
        lines = [item + '\n' for item in lines]
        # join the list back into a string and return it
        return ''.join(lines)

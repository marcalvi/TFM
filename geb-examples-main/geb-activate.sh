
export PATH=$PATH:/nfs/rwork/software/soft_ubuntu/3dslicer41120/Slicer-4.11.20210226-linux-amd64/


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/osiris-user/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/osiris-user/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/osiris-user/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/osiris-user/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#Change this
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"

# Activate the base enviroment by default.
conda activate base
#!/bin/bash
# set -ex

usage(){
echo "
You can run individual functions!

example:
  setup_demo
"
}

check_cd(){
  [ -d ".git" ] && return
  echo "Are you running $(basename $0) from the root path?"
  exit
}

is_sourced() {
  check_cd

  if [ -n "$ZSH_VERSION" ]; then 
      case $ZSH_EVAL_CONTEXT in *:file:*) return 0;; esac
  else  # Add additional POSIX-compatible shell names here, if needed.
      case ${0##*/} in dash|-dash|bash|-bash|ksh|-ksh|sh|-sh) return 0;; esac
  fi
  return 1  # NOT sourced.
}

setup_venv(){
  python3 -m venv venv
  source venv/bin/activate
  pip install -q -U pip

  check_venv || usage
}

check_venv(){
  # activate python venv
  [ -d venv ] && . venv/bin/activate || setup_venv
}


install_requirements(){
  [ -e requirements-devel.txt ] && \
    pip install -qr requirements-devel.txt
}

print_info(){
  echo "
  Run the following:

  # activate python virtual env (bash)
  source venv/bin/activate

  # blastoff w/ jupyter
  jupyter-lab
  "
}

setup_demo(){
  oc apply -k components
}

setup_demo(){
  check_cd
  check_venv
  install_requirements
  print_info
}

is_sourced && usage || setup_demo

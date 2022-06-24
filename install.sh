#!/bin/bash

enable=false ; if (( ${#@} == 1 || ${#@} ==  2 )) ; then enable=true ; fi

if $enable ;
then

    if (( ${#@} == 1 )) ;
    then
        requirements=$1

        if [[ "$requirements" == "yes" ]] ;
        then
        
            sudo apt-get install -y           python3 python3-pip
            sudo              ln -sf /usr/bin/python3 /usr/bin/py
            
            pip3 install nltk
            pip3 install numpy
            pip3 install utils
            pip3 install pandas
            pip3 install matplotlib
            
            # pip3 install pickle
            # pip3 install json
            
            pip3 install gensim
            pip3 install python-Levenshtein
            
            sudo apt install -y python3-sklearn
            # pip3 install random # already done

            py download_required_file.py
        
        else
            enable=false
        fi
            
    else
        dir=$1 ; file=$2 ;
        
        cd $dir ; py "$file".py
        cd ..
    fi
fi

if ! $enable ;
then
    echo ''
    echo 'arg 1 : install all requirements = [yes/..]'
    echo 'OR'
    echo 'arg 1 : Directory'
    echo 'arg 2 : File.R inside Directory'
    echo ''
fi

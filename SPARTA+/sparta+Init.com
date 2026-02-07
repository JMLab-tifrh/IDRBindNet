#!/bin/csh 

#
#
# Example usage:
#
#   if (-e /u/shenyang/com/sparta+Init.com) then
#      source /u/shenyang/com/sparta+Init.com
#   endif
#   

## PLEASE CHECK THE SPARTA+ INSTALLATION DIRECTORY DEFINED NEXT LINE!!!
setenv SPARTAP_DIR  /home/username/Softwares/sparta+/SPARTA+
setenv SPARTA_DIR  /home/username/Softwares/sparta+/SPARTA+

set path = (. $path ${SPARTAP_DIR})

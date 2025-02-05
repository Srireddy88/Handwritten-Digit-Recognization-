
#!/bin/bash

ehco "applicable for voting or not"
age1=18
age2=60
read age

if [[ $age -ge $age1 ]] && [[ $age -le $age2 ]]; then
	echo "applicable for voting"
else
	echo "not applicable for voting"
fi


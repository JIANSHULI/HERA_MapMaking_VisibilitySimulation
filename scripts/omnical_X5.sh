#! /bin/bash

#todofiles=`ls ~/fftt/calibration/fortran_code/logcal_todoq[0-5]?.txt`
#todofiles2=`ls ~/fftt/calibration/fortran_code/logcal_todoqC[0-5]?.txt`
#echo ${todofiles} ${todofiles2}
#for todofile in `echo ${todofiles} ${todofiles2}`; do
	#echo '========================================================================='
	#Q=`echo ${todofile} | awk -F"_todo" '{print $2}' | cut -d. -f1`
	#echo ${Q}
	#echo python ~/omnical/scripts/first_cal.py -p xx,yy `head ${todofile} -n 1` -t ${Q} -o ~/omnical/doc/ --overwrite
	#python ~/omnical/scripts/first_cal.py -p xx,yy `head ${todofile} -n 1` -t ${Q} -o ~/omnical/doc/ --overwrite
	#echo '-------------------------------------------------------------------------'
#done;

#heavy flagging
#q2C#q3A#qC3B#qC3A
#Q=q0C
#echo `date +%Y/%m/%d-%r` > ~/omnical4_$Q.log
#while read odfname
#do
    #echo "python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${Q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${Q}p.p --chemo 2 --chemot 3 --chemof 3 --flag --skip_sun"
    #echo "python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${Q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${Q}p.p --chemo 2 --chemot 3 --chemof 3 --flag --skip_sun" &>> ~/omnical4_$Q.log
    #python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${Q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${Q}p.p --chemo 2 --chemot 3 --chemof 3 --flag --skip_sun &>> ~/omnical4_$Q.log
#done < ~/fftt/calibration/fortran_code/logcal_todo${Q}.txt

#lenient flagging
q=q1A
Q=${q}L
echo `date +%Y/%m/%d-%r` > ~/omnical4_$Q.log
while read odfname
do
    echo "python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${q}p.p --chemo 1 --chemot 2 --chemof 2 --flag --skip_sun"
    echo "python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${q}p.p --chemo 1 --chemot 2 --chemof 2 --flag --skip_sun" &>> ~/omnical4_$Q.log
    python ~/omnical/scripts/omnical4.py  ${odfname} -t ${Q} -o /home/omniscope/data/X5/2015calibration/ -i /home/omniscope/omnical/doc/redundantinfo_first_cal_${q}p.bin -r /home/omniscope/omnical/doc/calpar_first_cal_${q}p.p --chemo 1 --chemot 2 --chemof 2 --flag --skip_sun &>> ~/omnical4_$Q.log
done < ~/fftt/calibration/fortran_code/logcal_todo${q}.txt

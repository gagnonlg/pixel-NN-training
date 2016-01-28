export tmp=$(mktemp -d)

calc_ROC () {
    db=$1
    name=$2
    sql_count_p=$3
    sql_count_n=$4
    sql_ROC=$5

    sqlite3 -separator " " $db "$sql_ROC" \
	| ./ROC $(sqlite3 $db "$sql_count_p") $(sqlite3 $db "$sql_count_n") \
	| ./ROC_Graph $name $tmp/$name.root
}
export -f calc_ROC

python2 gensql.py | parallel -P20 --colsep '\|' calc_ROC test/test.db 

hadd TEST_ROC.root $tmp/*.root

rm -r $tmp

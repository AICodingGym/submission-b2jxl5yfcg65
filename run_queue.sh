#!/bin/sh
# Run train_lstm.py jobs one after another, taking each job's arguments from the
# first line of queue.txt. Lines can be appended to queue.txt while this runs.
# Progress: queue_history.txt (START/END lines) and <run>.log for each job.
cd "$HOME/ventilator-pressure-prediction" || exit 1
while true; do
  line=$(head -n 1 queue.txt 2>/dev/null)
  [ -z "$line" ] && { echo "$(date '+%F %T') QUEUE EMPTY" >> queue_history.txt; break; }
  sed -i '1d' queue.txt
  run=$(echo "$line" | sed -E 's/.*--run ([^ ]+).*/\1/')
  echo "$(date '+%F %T') START $run :: $line" >> queue_history.txt
  python -u train_lstm.py $line > "$run.log" 2>&1
  echo "$(date '+%F %T') END $run exit=$?" >> queue_history.txt
done

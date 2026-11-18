"""Run the scripted supported-precrimp demo with live benchmark scoring."""
import sys
from finish_flap import main

if __name__=='__main__':
    options=['--engine','isaac','--physics-device','cpu','--finish-method','squeeze',
        '--finish-flaps','front,rear','--finish-goal-degrees','168','--finish-retained-degrees','150',
        '--squeeze-drive-n','40','--squeeze-force-n','37','--squeeze-hold-seconds','3',
        '--squeeze-hold-setting','full-close','--contact-alarm-ms','20',
        '--flap-sheet-mm','.3','--flap-damping','.03','--push-n','18','--initial-push-n','18',
        '--max-push-n','22','--flaps','front,rear','--degrees','175','--follow-live','--walk-contact',
        '--walk-distance-mm','3','--support-blend-mm','0','--staged-bends',
        '--brace-moment-at-com','--fold-seconds','25','--contact-speed-mm-s','5']
    sys.argv=[sys.argv[0],*options,*sys.argv[1:]]
    main()

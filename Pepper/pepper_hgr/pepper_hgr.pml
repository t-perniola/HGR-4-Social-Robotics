<?xml version="1.0" encoding="UTF-8" ?>
<Package name="pepper_hgr" format_version="5">
    <Manifest src="manifest.xml" />
    <BehaviorDescriptions>
        <BehaviorDescription name="behavior" src="applaud" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="kiss" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="heart" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="preghiera" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="dialog_test" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="voice_recog_test" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="dialogo" xar="behavior.xar" />
        <BehaviorDescription name="behavior" src="hello" xar="behavior.xar" />
    </BehaviorDescriptions>
    <Dialogs>
        <Dialog name="presentazione" src="presentazione/presentazione.dlg" />
    </Dialogs>
    <Resources>
        <File name="swiftswords_ext" src="applaud/swiftswords_ext.mp3" />
    </Resources>
    <Topics>
        <Topic name="presentazione_iti" src="presentazione/presentazione_iti.top" topicName="presentazione" language="it_IT" nuance="iti" />
    </Topics>
    <IgnoredPaths />
    <Translations auto-fill="en_US">
        <Translation name="translation_en_US" src="translations/translation_en_US.ts" language="en_US" />
        <Translation name="translation_it_IT" src="translations/translation_it_IT.ts" language="it_IT" />
    </Translations>
</Package>

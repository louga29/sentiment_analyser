#Requires AutoHotkey v2.0
#SingleInstance Force

choix := [",", ".", ":", "!", "?", "Verr. Maj"]
index := 1

MButton:: {
    majuscule := GetKeyState("Shift") != GetKeyState("CapsLock", "T")
    SendText majuscule ? "N" : "n"
}

XButton1:: {
    global index
    index := 1
    AfficherChoix()
    KeyWait "XButton1"
    ToolTip
    if (choix[index] = "Verr. Maj") {
        SetCapsLockState !GetKeyState("CapsLock", "T")
        ToolTip "Verr. Maj : " (GetKeyState("CapsLock", "T") ? "activé" : "désactivé")
        SetTimer () => ToolTip(), -800
    } else {
        SendText choix[index]
    }
}

#HotIf GetKeyState("XButton1", "P")
WheelDown::Changer(1)
WheelUp::Changer(-1)
#HotIf

Changer(pas) {
    global index
    index := Mod(index - 1 + pas + choix.Length, choix.Length) + 1
    AfficherChoix()
}

AfficherChoix() {
    texte := ""
    for i, c in choix
        texte .= (i = index ? "[ " c " ]" : "  " c "  ") "   "
    ToolTip texte
}

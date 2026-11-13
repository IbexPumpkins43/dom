"""The app's stylesheet. Colours come from the palette, so light and dark themes both work."""

STYLESHEET = """
#sidebar {
    background: palette(alternate-base);
    border-right: 1px solid palette(mid);
}
#sidebarHeading {
    color: palette(placeholder-text);
    font-size: 11px;
    font-weight: 600;
    padding: 6px 4px 0 4px;
}
#newChat {
    padding: 8px 12px;
    border: 1px solid palette(mid);
    border-radius: 8px;
    text-align: left;
    background: palette(base);
}
#newChat:hover { background: palette(midlight); }
#conversationList { background: transparent; outline: none; }
#conversationList::item { padding: 4px 8px; border-radius: 6px; }
#conversationList::item:hover { background: palette(midlight); }
#conversationList::item:selected {
    background: palette(highlight);
    color: palette(highlighted-text);
}

#topBar { border-bottom: 1px solid palette(midlight); }
#modelPicker { padding: 5px 10px; border-radius: 6px; }
#serverLabel { color: palette(placeholder-text); }
#serverLabel[state="failed"], #serverLabel[state="crashed"] { color: #d9534f; }

#banner { background: palette(alternate-base); border-bottom: 1px solid palette(mid); }
#banner[error="true"] { background: rgba(217, 83, 79, 0.15); border-bottom: 1px solid #d9534f; }

#chatArea, #chatCanvas { background: palette(base); }
#notice { color: palette(placeholder-text); font-size: 15px; padding: 60px 20px; }

#userBubble {
    background: palette(alternate-base);
    border: 1px solid palette(midlight);
    border-radius: 14px;
}
#assistantBubble { background: transparent; }
#thinkingToggle {
    border: none;
    color: palette(placeholder-text);
    font-size: 12px;
    padding: 0;
    background: transparent;
}
#thinkingText {
    color: palette(placeholder-text);
    font-size: 12px;
    border-left: 2px solid palette(mid);
    padding-left: 10px;
}

#composer {
    border: 1px solid palette(mid);
    border-radius: 14px;
    background: palette(base);
}
#chatInput { border: none; background: transparent; font-size: 14px; }
#sendButton {
    padding: 7px 18px;
    border-radius: 10px;
    background: palette(highlight);
    color: palette(highlighted-text);
    border: none;
    font-weight: 600;
}
#sendButton:disabled { background: palette(mid); color: palette(base); }
#sendButton[stop="true"] { background: #d9534f; color: white; }
#errorLabel { color: #d9534f; }
"""

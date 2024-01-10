#include <Arduino.h>        
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SSD1306_NO_SPLASH

// ----------------------------------------------------------------
//                         S E T T I N G S
// ----------------------------------------------------------------

    //  Arduino Mega
      #define encoder0PinA  2                   // Rotary encoder gpio pin
      #define encoder0PinB  3                   // Rotary encoder gpio pin
      #define encoder0Press 4                   // Rotary encoder button gpio pin
      #define OLEDC 0                           // oled clock pin (set to 0 for default)
      #define OLEDD 0                           // oled data pin
      #define OLEDE 0                           // oled enable pin

    // oLED
      #define OLED_ADDR 0x3C                    // OLED i2c address
      #define SCREEN_WIDTH 128                  // OLED display width, in pixels (usually 128)
      #define SCREEN_HEIGHT 64                  // OLED display height, in pixels (64 for larger oLEDs)
      #define OLED_RESET -1                     // Reset pin gpio (or -1 if sharing Arduino reset pin)

    // Misc
      const int serialDebug = 1;
      const int iLED = 2;                       // onboard indicator led gpio pin
      #define BUTTONPRESSEDSTATE 0              // rotary encoder gpio pin logic level when the button is pressed (usually 0)
      #define DEBOUNCEDELAY 20                  // debounce delay for button inputs
      const int menuTimeout = 10;               // menu inactivity timeout (seconds)
      const bool menuLargeText = 0;             // show larger text when possible (if struggling to read the small text)
      const int maxmenuItems = 12;              // max number of items used in any of the menus (keep as low as possible to save memory)
      const int itemTrigger = 2;                // rotary encoder - counts per tick (varies between encoders usually 1 or 2)
      const int topLine = 18;                   // y position of lower area of the display (18 with two colour displays)
      const byte lineSpace1 = 9;                // line spacing for textsize 1 (small text)
      const byte lineSpace2 = 17;               // line spacing for textsize 2 (large text)
      const int displayMaxLines = 5;            // max lines that can be displayed in lower section of display in textsize1 (5 on larger oLeds)
      const int MaxmenuTitleLength = 10;        // max characters per line when using text size 2 (usually 10)


// -------------------------------------------------------------------------------------------------
//                                 The custom menus go below here
// -------------------------------------------------------------------------------------------------

enum MenuItemType{
    DIAL,
    BOOL
};

struct DialMenuItem {
    int max_value;
    int min_malue;
    int step;
    int value;
};

struct BoolMenuItem {
    bool value;
};

union MenuItemData {
    DialMenuItem dialData;
    BoolMenuItem boolData;
};

struct MenuItem {
    String name;
    MenuItemType type;
    MenuItemData data;
};
 
// -------------------------------------------------------------------------------------------------
//                                         custom menus go above here
// -------------------------------------------------------------------------------------------------

enum menuModes {
      OFF,                                  // display is off
      MENU,                                 // a menu is active
      VALUE,                                // 'enter a value' none blocking is active
      MESSAGE,                              // displaying a message
      BLOCKING                              // a blocking procedure is in progress (see enter value)
  };
  menuModes menuMode = OFF;                 // default mode at startup is off

  struct Menu {
    // menu               // the title of active mode
    int noOfmenuItems = 0;                    // number if menu items in the active menu
    int highlightedMenuItem = 0;              // which item is curently highlighted in the menu
    MenuItem menuItems[maxmenuItems+1];         // store for the menu item titles
    uint32_t lastMenuActivity = 0;            // time the menu last saw any activity (used for timeout)
  };

  struct rotaryEncoders {
    volatile int encoder0Pos = 0;             // current value selected with rotary encoder (updated by interrupt routine)
    volatile bool encoderPrevA;               // used to debounced rotary encoder
    volatile bool encoderPrevB;               // used to debounced rotary encoder
    uint32_t reLastButtonChange = 0;          // last time state of button changed (for debouncing)
    bool encoderPrevButton = 0;               // used to debounce button
    int reButtonDebounced = 0;                // debounced current button state (1 when pressed)
    const bool reButtonPressedState = BUTTONPRESSEDSTATE;  // the logic level when the button is pressed
    const uint32_t reDebounceDelay = DEBOUNCEDELAY;        // button debounce delay setting
    bool reButtonPressed = 0;                 // flag set when the button is pressed (it has to be manually reset)
  };
  rotaryEncoders rotaryEncoder;

// oled SSD1306 display connected to I2C
  Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
 
// forward declarations
void doEncoder();
void reUpdateButton();
void selectItem();
void serviceMenu(Menu& menu);
int serviceValue(bool _blocking, Menu& menu);
void displayMessage(String _title, String _message);


Menu startMenu;
void createStartMenu() {
  menuMode = MENU;                      // enable menu mode     

  startMenu.noOfmenuItems = 9;

  startMenu.menuItems[1].name = "DIAL";
  startMenu.menuItems[1].type = DIAL;
  startMenu.menuItems[1].data.dialData = {50, 0, 1, 20};

  startMenu.menuItems[2].name = "BOOL";
  startMenu.menuItems[2].type = BOOL;
  startMenu.menuItems[2].data.boolData = {false};

  startMenu.menuItems[3].name = "BOOL";
  startMenu.menuItems[3].type = BOOL;
  startMenu.menuItems[3].data.boolData = {false};

  startMenu.menuItems[4].name = "BOOL";
  startMenu.menuItems[4].type = BOOL;
  startMenu.menuItems[4].data.boolData = {false};

  startMenu.menuItems[5].name = "BOOL";
  startMenu.menuItems[5].type = BOOL;
  startMenu.menuItems[5].data.boolData = {false};

  startMenu.menuItems[6].name = "BOOL";
  startMenu.menuItems[6].type = BOOL;
  startMenu.menuItems[6].data.boolData = {false};
  
  startMenu.menuItems[7].name = "BOOL";
  startMenu.menuItems[7].type = BOOL;
  startMenu.menuItems[7].data.boolData = {false};

  startMenu.menuItems[8].name = "BOOL";
  startMenu.menuItems[8].type = BOOL;
  startMenu.menuItems[8].data.boolData = {false};

  startMenu.menuItems[9].name = "BOOL";
  startMenu.menuItems[9].type = BOOL;
  startMenu.menuItems[9].data.boolData = {false};
}

Menu allMenus[1];
int currentMenu = 1;

void setup() {

  Serial.begin(115200); while (!Serial); delay(50);       // start serial comms
  Serial.println("\n\n\nStarting menu demo\n");

  pinMode(iLED, OUTPUT);     // onboard indicator led

  // configure gpio pins for rotary encoder
    pinMode(encoder0Press, INPUT_PULLUP);
    pinMode(encoder0PinA, INPUT);
    pinMode(encoder0PinB, INPUT);

  // initialise the oled display
    // enable pin
      if (OLEDE != 0) {
        pinMode(OLEDE , OUTPUT);
        digitalWrite(OLEDE, HIGH);
      }
    if (0 == OLEDC) Wire.begin();
    //else Wire.begin(OLEDD, OLEDC);
    if(!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
      if (serialDebug) Serial.println(("\nError initialising the oled display"));
    }
    Wire.setClock(100000);

  // Interrupt for reading the rotary encoder position
    rotaryEncoder.encoder0Pos = 0;
    attachInterrupt(digitalPinToInterrupt(encoder0PinA), doEncoder, CHANGE);

  //defaultMenu();       // start the default menu

  // display greeting message - pressing button will start menu
    displayMessage("STARTED", "BasicWebserver\nsketch");

    createStartMenu();
    Menu allMenus[1] = {startMenu};
}

void loop() {
  reUpdateButton();      // update rotary encoder button status (if pressed activate default menu)

  Menu& menu = allMenus[currentMenu];
  menuUpdate(menu);          // update or action the oled menu



  // flash onboard led
    static uint32_t ledTimer = millis();
    if ( (unsigned long)(millis() - ledTimer) > 500 ) {
      digitalWrite(iLED, !digitalRead(iLED));
      ledTimer = millis();
    }

}  // oledLoop



// ----------------------------------------------------------------
//                   -button debounce (rotary encoder)
// ----------------------------------------------------------------
// update rotary encoder current button status

void reUpdateButton() {
    bool tReading = digitalRead(encoder0Press);        // read current button state
    if (tReading != rotaryEncoder.encoderPrevButton) rotaryEncoder.reLastButtonChange = millis();     // if it has changed reset timer
    if ( (unsigned long)(millis() - rotaryEncoder.reLastButtonChange) > rotaryEncoder.reDebounceDelay ) {  // if button state is stable
      if (rotaryEncoder.encoderPrevButton == rotaryEncoder.reButtonPressedState) {
        if (rotaryEncoder.reButtonDebounced == 0) {    // if the button has been pressed
          rotaryEncoder.reButtonPressed = 1;           // flag set when the button has been pressed
          //if (menuMode == OFF) defaultMenu();          // if the display is off start the default menu
        }
        rotaryEncoder.reButtonDebounced = 1;           // debounced button status  (1 when pressed)
      } else {
        rotaryEncoder.reButtonDebounced = 0;
      }
    }
    rotaryEncoder.encoderPrevButton = tReading;            // update last state read
}



// ----------------------------------------------------------------
//                    -update the active menu
// ----------------------------------------------------------------

void menuUpdate(Menu& menu) {
    if (menuMode == OFF) return;    // if menu system is turned off do nothing more

    // if no recent activity then turn oled off
    if ( (unsigned long)(millis() - menu.lastMenuActivity) > (menuTimeout * 1000) ) {
        menuMode = OFF;
      return;
    }

    switch (menuMode) {

      // if there is an active menu
      case MENU:
        serviceMenu(menu);
        if (rotaryEncoder.reButtonPressed) {
            selectItem(menu);
        }
        break;

      // if there is an active none blocking 'enter value'
      case VALUE:
        serviceValue(0, menu);
        if (rotaryEncoder.reButtonPressed) {                        // if the button has been pressed
            menuMode = MENU;                                           // a value has been entered so action it
            rotaryEncoder.reButtonPressed = 0;
            break;
        }

      // if a message is being displayed
      case MESSAGE:
        //if (rotaryEncoder.reButtonPressed == 1) defaultMenu();    // if button has been pressed return to default menu
        break;
    }
}



// ----------------------------------------------------------------
//                       -service active menu
// ----------------------------------------------------------------

void serviceMenu(Menu& menu) {

    // rotary encoder
      if (rotaryEncoder.encoder0Pos >= itemTrigger) {
        rotaryEncoder.encoder0Pos -= itemTrigger;
        menu.highlightedMenuItem++;
        menu.lastMenuActivity = millis();   // log time
      }
      if (rotaryEncoder.encoder0Pos <= -itemTrigger) {
        rotaryEncoder.encoder0Pos += itemTrigger;
        menu.highlightedMenuItem--;
        menu.lastMenuActivity = millis();   // log time
      }

    const int _centreLine = displayMaxLines / 2 + 1;    // mid list point
    display.clearDisplay();
    display.setTextColor(WHITE);

    // verify valid highlighted item
      if (menu.highlightedMenuItem > menu.noOfmenuItems) menu.highlightedMenuItem = menu.noOfmenuItems;
      if (menu.highlightedMenuItem < 1) menu.highlightedMenuItem = 1;

    // menu
      display.setTextSize(2);
      display.setCursor(0, lineSpace1);
      for (int i=1; i <= displayMaxLines; i++) {
        int item = menu.highlightedMenuItem - _centreLine + i;
        if (item == menu.highlightedMenuItem) display.setTextColor(BLACK, WHITE);
        else display.setTextColor(WHITE);
        if (item > 0 && item <= menu.noOfmenuItems) display.println(menu.menuItems[item].name);
        else display.println(" ");
      }

    //// how to display some updating info. on the menu screen
    // display.setCursor(80, 25);
    // display.println(millis());
 
    display.display();
}

// ----------------------------------------------------------------
//                        -service value entry
// ----------------------------------------------------------------
// if _blocking set to 1 then all other tasks are stopped until a value is entered

int serviceValue(bool _blocking, Menu& menu) {

  const int _valueSpacingX = 30;      // spacing for the displayed value y position
  const int _valueSpacingY = 5;       // spacing for the displayed value y position

  if (_blocking) {
    menuMode = BLOCKING;
    menu.lastMenuActivity = millis();   // log time of last activity (for timeout)
  }
  uint32_t tTime;

  DialMenuItem& dialData = menu.menuItems[menu.highlightedMenuItem].data.dialData;

  do {

    // rotary encoder
      if (rotaryEncoder.encoder0Pos >= itemTrigger) {
        rotaryEncoder.encoder0Pos -= itemTrigger;
        dialData.value -= dialData.step;
        menu.lastMenuActivity = millis();   // log time
      }
      if (rotaryEncoder.encoder0Pos <= -itemTrigger) {
        rotaryEncoder.encoder0Pos += itemTrigger;
        dialData.value += dialData.step;
        menu.lastMenuActivity = millis();   // log time
      }
      if (dialData.value < dialData.min_malue) {
        dialData.value = dialData.min_malue;
        menu.lastMenuActivity = millis();   // log time
      }
      if (dialData.value > dialData.max_value) {
        dialData.value = dialData.max_value;
        menu.lastMenuActivity = millis();   // log time
      }

      display.clearDisplay();
      display.setTextColor(WHITE);

      // title
        display.setCursor(0, 0);
        if (menu.menuItems[menu.highlightedMenuItem].name.length() > MaxmenuTitleLength) display.setTextSize(1);
        else display.setTextSize(2);
        display.println(menu.menuItems[menu.highlightedMenuItem].name);
        display.drawLine(0, topLine-1, display.width(), topLine-1, WHITE);       // draw horizontal line under title

      // value selected
        display.setCursor(_valueSpacingX, topLine + _valueSpacingY);
        display.setTextSize(3);
        display.println(dialData.value);

      // range
        display.setCursor(0, display.height() - lineSpace1 - 1 );   // bottom of display
        display.setTextSize(1);
        display.println(String(dialData.min_malue) + " to " + String(dialData.max_value));

      // bar
        int Tlinelength = map(dialData.value, dialData.min_malue, dialData.max_value, 0 , display.width());
        display.drawLine(0, display.height()-1, Tlinelength, display.height()-1, WHITE);

      display.display();

      reUpdateButton();        // check status of button
      tTime = (unsigned long)(millis() - menu.lastMenuActivity);      // time since last activity

  } while (_blocking && rotaryEncoder.reButtonPressed == 0 && tTime < (menuTimeout * 1000));        // if in blocking mode repeat until button is pressed or timeout

  if (_blocking) menuMode = OFF;

  return dialData.value;        // used when in blocking mode

}


// ----------------------------------------------------------------
//                         -message display
// ----------------------------------------------------------------
// 21 characters per line, use "\n" for next line
// assistant:  <     line 1        ><     line 2        ><     line 3        ><     line 4         >

 void displayMessage(String _title, String _message) {
    display.clearDisplay();
    display.display();
    menuMode = MESSAGE;

    display.clearDisplay();
    display.setTextColor(WHITE);

    // title
    display.setCursor(0, 0);
    if (menuLargeText) {
      display.setTextSize(2);
      display.println(_title.substring(0, MaxmenuTitleLength));
    } else {
      if (_title.length() > MaxmenuTitleLength) display.setTextSize(1);
      else display.setTextSize(2);
      display.println(_title);
    }

    // message
    display.setCursor(0, topLine);
    display.setTextSize(1);
    display.println(_message);

    display.display();

 }

void doEncoder() {

  bool pinA = digitalRead(encoder0PinA);
  bool pinB = digitalRead(encoder0PinB);

  if ( (rotaryEncoder.encoderPrevA == pinA && rotaryEncoder.encoderPrevB == pinB) ) return;  // no change since last time (i.e. reject bounce)

  // same direction (alternating between 0,1 and 1,0 in one direction or 1,1 and 0,0 in the other direction)
         if (rotaryEncoder.encoderPrevA == 1 && rotaryEncoder.encoderPrevB == 0 && pinA == 0 && pinB == 1) rotaryEncoder.encoder0Pos -= 1;
    else if (rotaryEncoder.encoderPrevA == 0 && rotaryEncoder.encoderPrevB == 1 && pinA == 1 && pinB == 0) rotaryEncoder.encoder0Pos -= 1;
    else if (rotaryEncoder.encoderPrevA == 0 && rotaryEncoder.encoderPrevB == 0 && pinA == 1 && pinB == 1) rotaryEncoder.encoder0Pos += 1;
    else if (rotaryEncoder.encoderPrevA == 1 && rotaryEncoder.encoderPrevB == 1 && pinA == 0 && pinB == 0) rotaryEncoder.encoder0Pos += 1;

  // change of direction
    else if (rotaryEncoder.encoderPrevA == 1 && rotaryEncoder.encoderPrevB == 0 && pinA == 0 && pinB == 0) rotaryEncoder.encoder0Pos += 1;
    else if (rotaryEncoder.encoderPrevA == 0 && rotaryEncoder.encoderPrevB == 1 && pinA == 1 && pinB == 1) rotaryEncoder.encoder0Pos += 1;
    else if (rotaryEncoder.encoderPrevA == 0 && rotaryEncoder.encoderPrevB == 0 && pinA == 1 && pinB == 0) rotaryEncoder.encoder0Pos -= 1;
    else if (rotaryEncoder.encoderPrevA == 1 && rotaryEncoder.encoderPrevB == 1 && pinA == 0 && pinB == 1) rotaryEncoder.encoder0Pos -= 1;

  //else if (serialDebug) Serial.println("Error: invalid rotary encoder pin state - prev=" + String(rotaryEncoder.encoderPrevA) + ","
  //                                      + String(rotaryEncoder.encoderPrevB) + " new=" + String(pinA) + "," + String(pinB));

  // update previous readings
    rotaryEncoder.encoderPrevA = pinA;
    rotaryEncoder.encoderPrevB = pinB;
}

void selectItem(Menu& menu) {
    menu.lastMenuActivity = millis();
    if (serialDebug) Serial.println("' item '" + menu.menuItems[menu.highlightedMenuItem].name + "' selected");
    MenuItem selectedMenuItem = menu.menuItems[menu.highlightedMenuItem];
    switch(selectedMenuItem.type) {
        case DIAL:
            menuMode = VALUE;
            break;
        case BOOL:
            break;
    }
    rotaryEncoder.reButtonPressed = 0;
}